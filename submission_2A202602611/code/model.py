"""model.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm/class có `raise NotImplementedError`.

Model: MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()  # Khởi tạo phần quản lý lớp và tham số mà nn.Module cung cấp.
        layers = []  # Danh sách chứa các lớp theo thứ tự dữ liệu sẽ đi qua.
        previous_features = in_features  # Lớp đầu tiên nhận 54 giá trị của mỗi mẫu theo mặc định.
        for features in hidden:  # Lần lượt lấy số nơ-ron của mỗi lớp ẩn, ví dụ 256 rồi 128.
            layers.append(nn.Linear(previous_features, features))  # Tạo lớp tính đầu ra = đầu vào × trọng số + bias (giá trị cộng thêm).
            layers.append(nn.ReLU())  # Giữ giá trị dương và đổi giá trị âm thành 0.
            if dropout > 0.0:  # Chỉ thêm lớp tắt ngẫu nhiên khi xác suất tắt lớn hơn 0.
                layers.append(nn.Dropout(dropout))  # Khi huấn luyện, tắt ngẫu nhiên một phần giá trị và điều chỉnh các giá trị còn lại.
            previous_features = features  # Số đầu ra của lớp này trở thành số đầu vào của lớp kế tiếp.
        layers.append(nn.Linear(previous_features, num_classes))  # Lớp cuối trả về điểm số cho 7 loại rừng theo mặc định.
        self.net = nn.Sequential(*layers)  # *layers đưa từng lớp trong danh sách vào một mạng chạy tuần tự.
        init_weights(self, init)  # Đặt giá trị ban đầu cho trọng số theo cách được chọn qua init.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)  # Cho x đi qua toàn bộ mạng và trả về điểm số của các lớp, chưa đổi thành xác suất.


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0).

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out)); nếu bạn dùng Var = 1/n_in theo slide, hãy ghi rõ
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in)
        "default" : không làm gì (giữ khởi tạo mặc định của nn.Linear; KHÔNG phải He)
    Gợi ý: duyệt model.modules(), chọn isinstance(m, nn.Linear).
    """
    if init not in {"zeros", "normal", "xavier", "he", "default"}:  # Kiểm tra tên cách khởi tạo có được hỗ trợ không.
        raise ValueError(f"Unknown initialization: {init!r}")  # Báo lỗi kèm giá trị đã nhập; !r giúp hiện rõ chuỗi đó.
    if init == "default":  # Nếu chọn mặc định, giữ nguyên các giá trị do nn.Linear tạo sẵn.
        return  # Kết thúc hàm, không thay đổi cả trọng số lẫn bias.

    for layer in model.modules():  # Duyệt mạng và tất cả các lớp bên trong mạng.
        if isinstance(layer, nn.Linear):  # Chỉ xử lý lớp Linear vì lớp này chứa trọng số và bias.
            if init == "zeros":  # Cách zeros: mọi trọng số bắt đầu bằng 0.
                nn.init.zeros_(layer.weight)  # Ghi số 0 vào toàn bộ trọng số của lớp hiện tại.
            elif init == "normal":  # Cách normal: chọn ngẫu nhiên các trọng số nhỏ quanh 0.
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # Trung bình bằng 0; độ lệch chuẩn 0.01 cho biết mức phân tán quanh 0.
            elif init == "xavier":  # Cách Xavier: chọn mức phân tán dựa trên cả số đầu vào và đầu ra.
                nn.init.xavier_normal_(layer.weight)  # Điền trọng số ngẫu nhiên theo cách Xavier.
            elif init == "he":  # Cách He: chọn mức phân tán phù hợp với lớp dùng ReLU.
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")  # Điền trọng số theo cách He với thiết lập dành cho ReLU.
            if layer.bias is not None:  # Kiểm tra lớp có phần giá trị cộng thêm hay không.
                nn.init.zeros_(layer.bias)  # Đặt phần cộng thêm bằng 0 cho các cách khởi tạo ở trên.


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS ngay sau khi tạo model."""
    # model.parameters(): lấy các trọng số và bias của model.
    # requires_grad: chỉ đếm tham số được phép cập nhật khi huấn luyện.
    # numel(): đếm số phần tử trong mỗi tham số; sum(): cộng tất cả lại.
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau mỗi lớp (ở bước 0, một lô val) — dùng cho thí nghiệm khởi tạo.

    Các bước:
      1. model.eval(); h = x
      2. duyệt từng lớp con theo thứ tự; sau mỗi nn.Linear (hoặc sau mỗi ReLU, bạn chọn và ghi rõ) lưu h.std().item()
      3. trả về danh sách std theo lớp
    """
    raise NotImplementedError  # TODO
