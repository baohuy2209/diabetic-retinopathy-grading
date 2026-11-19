import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
NAMES=['01_E0_decision_optimization_resume_v2.ipynb','02_E1_representation_probes_resume_v2.ipynb']


def namespace(stage):
    doc=json.loads((ROOT/'notebooks'/'mlwa'/NAMES[stage]).read_text('utf-8'))
    ns={'__name__':'resume_notebook_test','display':lambda value:None}
    for index, cell in enumerate(doc['cells']):
        code = ''.join(cell['source'])
        if cell['cell_type'] != 'code' or not code.startswith(('NOTEBOOK_CODE_SHA256', 'ANALYSIS_PLAN_SHA256', '"""Shared definitions', '"""Transactional CPU', '"""Source-selected', '"""Frozen encoder probes')):
            continue
        exec(compile(''.join(doc['cells'][index]['source']),f'notebook-cell-{index}','exec'),ns)
    return ns


def make_p0(ns,root,verified=False):
    rng=np.random.default_rng(24)
    for variant in ['V2','V3']:
        for split in ['source_train','source_val','target']:
            y=np.tile(np.arange(5),8)
            ids=[f'{split}_image{i:03d}' for i in range(len(y))]
            p=rng.dirichlet(np.ones(5),len(y))
            df=pd.DataFrame(p,columns=ns['PROB_COLS'])
            df['image_key']=ids;df['image_id']=ids;df['y_true']=y
            df['patient_key']=[f'{split}_patient{i//2}' for i in range(len(y))]
            df['patient_identity_status']='verified' if verified else 'legacy_unverified'
            df['variant']=variant;df['fold_id']=0;df['split_role']=split
            df['cohort']='aptos' if split=='target' else np.where(np.arange(len(y))%2,'idrid','ddr')
            df['pred_native']=p.argmax(1)
            df['s']=y+rng.normal(0,.04,len(y)) if variant=='V3' else np.nan
            for k in range(4):df[f'theta{k}']=k+.5 if variant=='V3' else np.nan
            z=np.tile(np.eye(5)[y],(1,103))[:,:512]+rng.normal(0,.02,(len(y),512))
            folder=root/'lodo'/f'{variant}_fold_0_aptos'/split
            folder.mkdir(parents=True)
            df.to_csv(folder/'predictions.csv',index=False)
            np.save(folder/'z.npy',z.astype(np.float32))
            # A redundant part is present to ensure E0/E1 do not stage it again.
            (folder/'parts').mkdir();(folder/'parts'/'redundant.bin').write_bytes(b'not required by consumers')
            identity=dict(variant=variant,fold=0,split=split,synthetic_TEST_ONLY=True,
                checkpoint_sha256='test_checkpoint_'+variant,code_sha256='synthetic_test_code',forward_id='TEST_ONLY',
                manifest_sha256='test_manifest_'+split,precision='fp32',patch_cap=256,patch_chunk=32,
                data_archive_sha256='TEST_ONLY',diagnostic_limit=0)
            ns['complete_directory'](folder,identity,ids)
    return root


def test_notebook_schema_compiles_and_hashes():
    import ast,nbformat
    for name in NAMES:
        doc=nbformat.read(ROOT/'notebooks'/'mlwa'/name,as_version=4);nbformat.validate(doc)
        sources=[]
        for c in doc.cells:
            if c.cell_type!='code':continue
            assert c.execution_count is None and not c.outputs
            tree=ast.parse(c.source);compile(tree,'<cell>','exec')
            if not c.source.startswith('NOTEBOOK_CODE_SHA256'):sources.append(c.source)
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):assert not(node.module or '').startswith(('src','mlwa_experiments'))
            assert 'shutil.copytree' not in c.source
        assert hashlib.sha256('\n'.join(sources).encode()).hexdigest()==doc.metadata.mlwa.code_sha256


def test_units_reuse_across_reset_and_reject_mismatch(tmp_path):
    ns=namespace(0);calls=[]
    def producer(folder):calls.append(1);return {'value':7},{'x':np.arange(5)}
    cache=ns['AnalysisCache'](tmp_path/'runtime1',tmp_path/'drive','code')
    _,a,_=cache.unit('step',{'seed':42},producer)
    reset=ns['AnalysisCache'](tmp_path/'runtime2',tmp_path/'drive','code')
    _,b,_=reset.unit('step',{'seed':42},producer)
    assert len(calls)==1;np.testing.assert_array_equal(a['x'],b['x'])
    with pytest.raises(ValueError,match='changed'):
        reset.unit('step',{'seed':43},producer)
    (tmp_path/'drive/units/step/arrays.npz').write_bytes(b'corrupt')
    third=ns['AnalysisCache'](tmp_path/'runtime3',tmp_path/'drive','code')
    with pytest.raises(ValueError,match='Corrupt'):
        third.unit('step',{'seed':42},producer)


def test_incomplete_unit_recomputed_and_completed_unit_preserved(tmp_path):
    ns=namespace(0);cache=ns['AnalysisCache'](tmp_path/'local',tmp_path/'drive','code')
    cache.unit('first',{},lambda folder:({'ok':True},{}))
    def interrupt(folder):
        (folder/'intermediate.tmp').write_text('partial')
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):cache.unit('second',{},interrupt)
    assert not(tmp_path/'drive/units/second/COMPLETE.json').exists()
    reset=ns['AnalysisCache'](tmp_path/'reset',tmp_path/'drive','code')
    def must_not_run(folder):raise AssertionError('Completed unit recomputed')
    reset.unit('first',{},must_not_run)
    reset.unit('second',{},lambda folder:({'ok':True},{}))
    assert reset.hits==1 and reset.computed==1


def test_staging_only_core_inputs_no_p0_mutations(tmp_path):
    ns=namespace(0);source=make_p0(ns,tmp_path/'P0')
    before={str(p.relative_to(source)):ns['sha256_file'](p) for p in source.rglob('*') if p.is_file()}
    ns['stage_p0_inputs'](source,tmp_path/'E0_inputs',[0],['V2','V3'],False)
    assert not list((tmp_path/'E0_inputs').rglob('z.npy'))
    assert not list((tmp_path/'E0_inputs').rglob('redundant.bin'))
    ns['stage_p0_inputs'](source,tmp_path/'E1_inputs',[0],['V2','V3'],True)
    assert len(list((tmp_path/'E1_inputs').rglob('z.npy')))==6
    ns['stage_p0_inputs'](source,tmp_path/'E0_inputs',[0],['V2','V3'],True)
    assert len(list((tmp_path/'E0_inputs').rglob('z.npy')))==6
    after={str(p.relative_to(source)):ns['sha256_file'](p) for p in source.rglob('*') if p.is_file()}
    assert before==after
    folder=source/'lodo/V3_fold_0_aptos/target'
    marker=json.loads((folder/'COMPLETE.json').read_text());marker['identity']['checkpoint_sha256']='wrong'
    ns['write_json'](folder/'COMPLETE.json',marker)
    with pytest.raises(ValueError,match='Mixed'):
        ns['stage_p0_inputs'](source,tmp_path/'fresh_inputs',[0],['V3'],True)


def test_e0_full_integration_resume_and_math(tmp_path):
    ns=namespace(0);p0=make_p0(ns,tmp_path/'P0',verified=True)
    table,cache=ns['run_e0_resumable'](p0,tmp_path/'input',tmp_path/'local',tmp_path/'drive',
                                    [0],['V2','V3'],bootstrap_reps=6,stability_reps=4,bootstrap_batch=2)
    assert len(table)==22
    completed=len(list((tmp_path/'drive').rglob('COMPLETE.json')))
    assert completed>10
    # Cell/code math parity: selecting the same source data returns exactly the original chosen config.
    source,_=ns['load_shard'](tmp_path/'input','V3',0,'source_val')
    train,_=ns['load_shard'](tmp_path/'input','V3',0,'source_train')
    chosen,_=ns['select_cibd'](source,ns['prior_from_labels'](train.y_true))
    selection=json.loads((tmp_path/'drive/units/V3_fold_0_aptos/source_selection/result.json').read_text())
    assert selection['selected']==chosen
    ns['select_cibd']=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Recomputed completed E0'))
    repeat,cached=ns['run_e0_resumable'](p0,tmp_path/'reset_input',tmp_path/'reset_local',tmp_path/'drive',
                                    [0],['V2','V3'],bootstrap_reps=6,stability_reps=4,bootstrap_batch=2)
    assert cached.computed==0 and len(repeat)==len(table)
    pd.testing.assert_frame_equal(table,repeat,check_dtype=False)


def test_e1_interrupted_fit_resume_across_fresh_runtime(tmp_path):
    ns=namespace(1);p0=make_p0(ns,tmp_path/'P0',verified=True)
    original=ns['fit_logistic'];count={'calls':0}
    def interrupt_third(*args,**kwargs):
        count['calls']+=1
        if count['calls']==3:raise KeyboardInterrupt
        return original(*args,**kwargs)
    ns['fit_logistic']=interrupt_third
    with pytest.raises(KeyboardInterrupt):
        ns['run_e1_resumable'](p0,tmp_path/'input1',tmp_path/'local1',tmp_path/'drive',[0],
                              [.01,1],[0,1,2],bootstrap_reps=6,bootstrap_batch=2,cpu_threads=1)
    assert (tmp_path/'drive/units/fold_0_aptos/z_v2/C_0.01/COMPLETE.json').exists()
    assert (tmp_path/'drive/units/fold_0_aptos/z_v2/C_1/COMPLETE.json').exists()
    ns['fit_logistic']=original
    table,cache=ns['run_e1_resumable'](p0,tmp_path/'input2',tmp_path/'local2',tmp_path/'drive',[0],
                              [.01,1],[0,1,2],bootstrap_reps=6,bootstrap_batch=2,cpu_threads=1)
    assert cache.hits>=2 and len(table)==39
    ns['fit_logistic']=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Refit completed E1'))
    repeat,cached=ns['run_e1_resumable'](p0,tmp_path/'input3',tmp_path/'local3',tmp_path/'drive',[0],
                              [.01,1],[0,1,2],bootstrap_reps=6,bootstrap_batch=2,cpu_threads=1)
    assert cached.computed==0
    pd.testing.assert_frame_equal(table,repeat,check_dtype=False)


def test_linear_state_restore_exact_predictions_and_middle_array_hash():
    ns=namespace(1);rng=np.random.default_rng(2);y=np.tile(np.arange(5),10)
    x=np.eye(5)[y]+rng.normal(0,.02,(50,5))
    scaler=ns['StandardScaler']().fit(x);model,ok=ns['fit_logistic'](scaler.transform(x),y,1)
    state=ns['pack_linear_state'](model,scaler)
    restored,scaled=ns['unpack_linear_state']({'C':1},state)
    np.testing.assert_allclose(model.predict_proba(scaler.transform(x)),restored.predict_proba(scaled.transform(x)),rtol=0,atol=0)
    a=np.zeros(10000);b=a.copy();b[5000]=1
    assert ns['array_hash'](a)!=ns['array_hash'](b)


def test_patient_bootstrap_batch_resume_and_missing_identity(tmp_path):
    ns=namespace(0);y=np.tile(np.arange(5),8)
    frame=pd.DataFrame(dict(y_true=y,patient_key=[f'p{i//2}' for i in range(40)],patient_identity_status='verified'))
    a=y.copy();b=y.copy();b[::3]=0
    cache=ns['AnalysisCache'](tmp_path/'local',tmp_path/'drive','code')
    first=ns['cached_pair_ci'](cache,'pair',frame,a,b,['recall_g3','within1'],6,2,{'input':'x'})
    reset=ns['AnalysisCache'](tmp_path/'reset',tmp_path/'drive','code')
    second=ns['cached_pair_ci'](reset,'pair',frame,a,b,['recall_g3','within1'],6,2,{'input':'x'})
    assert first==second and reset.computed==0 and reset.hits==3
    frame['patient_identity_status']='legacy_unverified'
    missing=ns['cached_pair_ci'](cache,'missing',frame,a,b,['recall_g3'],6,2,{})
    assert missing[0]['status']=='NOT_ESTIMABLE' and np.isnan(missing[0]['lower'])


def test_source_only_fit_api_and_permutation():
    ns=namespace(1);rng=np.random.default_rng(8);y=np.tile(np.arange(5),8)
    x=np.eye(5)[y]+rng.normal(0,.05,(40,5))
    scaler=ns['StandardScaler']().fit(x)
    model,_=ns['fit_logistic'](scaler.transform(x),y,1)
    before=model.coef_.copy();mean=scaler.mean_.copy()
    # Extreme target transforms cannot refit train scaler/coefficients.
    model.predict_proba(scaler.transform(np.full((8,5),999)))
    np.testing.assert_array_equal(model.coef_,before);np.testing.assert_array_equal(scaler.mean_,mean)
    frame=pd.DataFrame(dict(cohort=['a']*20+['b']*20))
    for seed in [0,1,2]:
        ix=ns['permutation_by_cohort'](frame,seed)
        assert set(ix[:20])==set(range(20)) and set(ix[20:])==set(range(20,40))
        np.testing.assert_array_equal(np.bincount(y[:20]),np.bincount(y[ix[:20]]))


def test_key_paper_cost_special_cases_and_posterior_invariance():
    ns=namespace(0);rng=np.random.default_rng(44);p=rng.dirichlet(np.ones(5),100)
    prior=np.array([.6,.1,.1,.1,.1]);y=np.arange(100)%5
    np.testing.assert_array_equal(ns['cibd'](p,prior,0,0,1),p.argmax(1))
    np.testing.assert_array_equal(ns['cibd'](p,prior,0,1,2),(p/prior).argmax(1))
    np.testing.assert_array_equal(ns['cibd'](p,prior,1,0,1),(p.cumsum(1)>=.5).argmax(1))
    np.testing.assert_array_equal(ns['cibd'](p,prior,1,0,2),np.ceil(p@np.arange(5)-.5).clip(0,4).astype(int))
    frame=pd.DataFrame(p,columns=ns['PROB_COLS']);frame['pred_native']=p.argmax(1);frame['s']=np.nan
    original=p.copy();quality=ns['posterior_metrics'](y,p)
    ns['decode_all'](frame,prior,dict(alpha=.5,tau=.5,q=2,fallback=False))
    np.testing.assert_array_equal(p,original)
    assert ns['posterior_metrics'](y,p)==quality
    assert np.isclose(quality['nll_mean']*len(y),-np.log(p[np.arange(len(y)),y]).sum())


def test_epstein_reward_is_one_minus_normalized_rps_loss():
    ns=namespace(0);rng=np.random.default_rng(8);p=rng.dirichlet(np.ones(5),40);y=np.arange(40)%5
    # Epstein 1969 p987 recommended S_j, indices shifted to 0..4.
    cdf=p.cumsum(1)[:,:4]
    reward=1.5-np.sum(cdf**2+(1-cdf)**2,axis=1)/8-np.sum(abs(np.arange(5)[None,:]-y[:,None])*p,axis=1)/4
    assert np.isclose(ns['posterior_metrics'](y,p)['rps_loss'],np.mean(1-reward))
    perfect=ns['posterior_metrics'](np.arange(5),np.eye(5))
    assert perfect['rps_loss']==0 and perfect['ece_15']==0


def test_source_screening_and_cloglog_median_offset():
    ns=namespace(0)
    y=np.r_[np.zeros(100,int),np.full(20,3)]
    risk=np.r_[np.linspace(0,.7,100),np.linspace(.3,.9,20)]
    p=np.zeros((120,5));p[:,0]=1-risk;p[:,2]=risk
    threshold=ns['screening_threshold'](y,p)
    metrics=ns['screening_metrics'](y,p,threshold)
    assert metrics['specificity']>=.95 and metrics['sensitivity']<.95
    theta=np.array([-1.5,-.47,.60,1.55]);boundary=theta[2]-np.log(np.log(2))
    s=np.array([boundary-1e-6,boundary+1e-6])
    cdf=1-np.exp(-np.exp(theta[None,:]-s[:,None]))
    pmf=np.column_stack([cdf[:,0],np.diff(cdf,axis=1),1-cdf[:,-1]])
    np.testing.assert_array_equal((pmf.cumsum(1)>=.5).argmax(1),[2,3])


def test_reject_missing_p0_completion_and_wrong_output_tree(tmp_path):
    ns=namespace(0);p0=make_p0(ns,tmp_path/'P0')
    with pytest.raises(ValueError,match='separate'):
        ns['separate_paths'](p0,p0/'analysis')
    (p0/'lodo/V2_fold_0_aptos/source_val/COMPLETE.json').unlink()
    with pytest.raises(ValueError,match='not complete'):
        ns['stage_p0_inputs'](p0,tmp_path/'inputs',[0],['V2'],False)
