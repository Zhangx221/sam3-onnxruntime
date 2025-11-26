import onnx
from onnx import helper, TensorProto
import os

ori_model = "models/onnx_detect/sam3.onnx"
modify_model = "models/onnx_detect_modify/sam3.onnx"

os.makedirs(os.path.dirname(modify_model), exist_ok=True)

model = onnx.load(ori_model)

# 1. 找到 /sam3/If 节点
if_node = None
for node in model.graph.node:
    if node.name == "/sam3/If":
        if_node = node
        break

assert if_node is not None, "没找到 /sam3/If 节点"

# 2. 取出 then_branch / else_branch 子图
then_branch = None
else_branch = None
for attr in if_node.attribute:
    if attr.name == "then_branch":
        then_branch = attr.g
    elif attr.name == "else_branch":
        else_branch = attr.g

assert then_branch is not None and else_branch is not None

# 3. 在 then_branch 子图里加 Unsqueeze 节点
unsq_node = helper.make_node(
    "Unsqueeze",
    inputs=["/sam3/Squeeze_4_output_0"],          # Squeeze 的原输出
    outputs=["/sam3/Squeeze_4_unsq_output_0"],    # 新的 4D 输出
    name="/sam3/Squeeze_4_unsq",
    axes=[-1],  # 在最后一维加 dim=1
)
then_branch.node.append(unsq_node)

# 4. 修改 then_branch 的输出 value_info，让它输出新的名字 + 4D shape
then_out = then_branch.output[0]
then_out.name = "/sam3/Squeeze_4_unsq_output_0"


tensor_type = then_out.type.tensor_type
shape = tensor_type.shape

# shape.dim 原来有 3 个 dim: 6,1,200
# 给它再加一个 dim=1
if len(shape.dim) == 3:
    new_dim = shape.dim.add()
    new_dim.dim_value = 1  # 最后一维 = 1

# 5. 保存（注意大模型用 external_data）
onnx.save_model(
    model,
    modify_model,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="sam3.data",
    size_threshold=1024,
)
