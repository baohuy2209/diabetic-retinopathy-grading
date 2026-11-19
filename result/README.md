# Đưa log kết quả vào repo (không chạy lại)

Bạn chỉ cần chép các file đã có vào cấu trúc dưới đây. Không cần checkpoint, ảnh fundus hoặc chạy lại bất kỳ thí nghiệm nào để lưu hồ sơ kết quả.

```text
result/
  README.md
  reference/
    lodo_summary_with_calibrated_screening.json
  published/
    <ten-bo-ket-qua-goc>/
      <giu nguyen cac CSV, JSON, TEX, PDF va PNG goc>
  diagnostic/
    <run_id>/
      <giu nguyen cau truc lan chay chan doan>
  logs/
    <run_id>/
      execution.log
      colab_config.json
      reports/
        provenance.json
  manifests/
    lodo_folds/
      fold_0_aptos_test.csv
      fold_1_idrid_test.csv
      fold_2_messidor2_test.csv
      fold_3_ddr_test.csv
      fold_4_eyepacs_test.csv
      fold_5_deepdrid_test.csv
  raw_preds/                         # Tuỳ chọn, để tính lại CI sau này
    V2_PlusGlobalContext_H2__aptos.npz
    V3_PlusOrdinalHead_H3_Champion__aptos.npz
    ...                             # Tổng cộng 12 NPZ cho 6 cohort
```

## 1. File tham chiếu

Chép file gốc sau vào `result/reference/`, giữ nguyên tên và nội dung:

```text
D:/Github/deep-learning-for-computer-vision/notebooks/dmip-conference/diabetic_retinopathy/pipeline/results/lodo_summary_with_calibrated_screening.json
```

Đây là bảng tham chiếu của benchmark, không phải log chứng minh một lần chạy mới đã tái lập thành công.

## 2. Bộ kết quả dùng trong bài

Trong `result/published/<ten-bo-ket-qua-goc>/`, đưa các đầu ra đã được xác định là nguồn số liệu của manuscript:

- Bảng chỉ số sáu fold: QWK, Within-1, Grade-3 sensitivity và số mẫu.
- Bảng paired bootstrap: CI cho V2, V3 và V3−V2, kèm số lần resample và seed.
- Grade-3 confusion theo cohort và tổng hợp: đủ hướng G3→0/1/2/3/4.
- Các bảng LaTeX và hình tương ứng nếu đã có.
- Báo cáo provenance/audit đi kèm, ghi rõ nguồn là predictions gốc hay inference mới.

Giữ nguyên các tên như `grade3_bootstrap_by_cohort.csv`, `grade3_confusion_pooled.csv` hoặc các tên có hậu tố trạng thái mà công cụ gốc đã tạo. Không đổi tên file `DIAGNOSTIC` thành `VERIFIED` hay xoá thông tin run.

**Không chép cả `camera_ready_analysis/results/` rồi mặc định tất cả đều là kết quả cuối.** Thư mục phát triển chứa cả đầu ra thử, đầu ra cũ và đầu ra chẩn đoán. Cần chọn đúng bộ gốc đã đối chiếu với bài, giữ nguyên từng thư mục run thay vì trộn các file khác phiên.

## 3. Log của phiên chạy

Từ thư mục phiên Colab hoặc `camera_ready_analysis/runs/`, chép nguyên bộ log vào `result/logs/<run_id>/`: `execution.log`, `colab_config.json`, `reports/provenance.json` và các báo cáo kiểm tra kèm theo nếu phiên đó có tạo. Nếu tên hoặc cấu trúc gốc khác, giữ nguyên cấu trúc gốc. File nào không có thì ghi rõ thiếu, không tự dựng log hoặc bổ sung hash lịch sử.

Các phiên inference mới chưa khớp benchmark nằm trong `result/diagnostic/<run_id>/`; không dùng chúng để thay thế kết quả gốc trong `published/`. Một cache có hash hợp lệ vẫn có thể khác chỉ số của manuscript.

## 4. Predictions và manifests (tuỳ chọn)

Log tổng hợp đủ để lưu kết quả; để người khác tính lại paired bootstrap, cần thêm **12 NPZ predictions gốc** và **6 test manifests đúng phiên**. Các NPZ phải giữ nguyên mảng dự đoán; không dựng lại từ QWK, bảng confusion hoặc CI. Nếu NPZ thiếu image ID, giữ tài liệu chứng minh loader không shuffle và nêu rõ pairing dựa trên thứ tự manifest là một giả định lịch sử, không phải bằng chứng độc lập về danh tính từng ảnh.

Bạn có thể chép thêm train/val manifests để ghi nhận cách chia tập. Giữ số dòng thực tế của log/manifests, không sửa thành tổng số ảnh nghiên cứu. Không tự gán nguyên nhân loại ảnh Messidor-2 nếu log lịch sử không chứng minh.

NPZ mặc định bị `.gitignore` loại vì có thể lớn. Nếu muốn đưa vào Git và được phép chia sẻ, thêm đích danh từng file, ví dụ:

```powershell
git add -f result/raw_preds/V2_PlusGlobalContext_H2__aptos.npz
```

Không dùng lệnh thêm cưỡng bức cả thư mục dữ liệu/checkpoint. CSV/JSON/log/bảng/hình dưới `result/` được phép theo dõi; kiểm tra đường dẫn máy và thông tin riêng trong log trước khi công khai.

## 5. Kiểm kê file mà không chạy thí nghiệm

Sau khi chép xong, tại thư mục repo:

```powershell
python -m experiments.audit_bundle --input-dir result/published
```

Lệnh này chỉ đọc file và in danh sách cùng SHA-256. Nó không nạp checkpoint, không inference, không bootstrap và không thay đổi log. `INVENTORIED_ONLY` chỉ xác nhận đã kiểm kê, không chứng nhận số liệu đúng. Không đặt file trọng số `.pth/.pt/.ckpt/.safetensors` trong bộ kết quả.
