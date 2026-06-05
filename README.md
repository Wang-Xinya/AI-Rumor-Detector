# 2026《人工智能导论》大作业 - 可解释的谣言检测

## 1. 主要目标与要求

**目标**：对社交媒体推文进行二分类（谣言 / 非谣言），同时输出一段自然语言解释判断依据

**重要要求**：
- 准确率尽量高（按照wsl老师的说法，正确率最好达到80%以上），解释需连贯、可读
- 模型需能运行（已实现）
- 最终提交：GitHub 仓库（包含报告PDF和代码）

## 2. 文件内容与作用

| 文件 | 作用 | 主要函数 |
|------|------|----------|
| `harness_base.py` | 定义 Harness 基类，提供 `update` / `predict` 接口 | `Harness.__init__`, `update`, `predict` |
| `llm_client.py` | 封装学校 LLM API 调用 | `call_llm(messages)` → 返回字符串 |
| `detection.py` | 核心分类逻辑，继承 Harness | `update()`（构建 BM25 索引 + 精确匹配表）<br>`predict()`（BM25 检索 + 构造 prompt + 调用 LLM + 解析）<br>`get_last_explanation()`（获取解释） |
| `run.py` | 加载数据、训练、评估、保存结果 | `load_data()` → 读取 CSV<br>`evaluate()` → 循环预测并计算指标<br>`main()` → 整体流程 |
| `requirements.txt` | 依赖库 | - |

## 3. 当前版本的主要策略与思路

**整体策略**：BM25 检索 + Few-shot 推理 + 强制输出解释
（人话：在训练集中检索与目标样本相似的样本提供给 LLM，帮助判断）

1. **训练阶段（`update`）**  
   - 存储每条样本的原始文本、标签（0/1）  
   - 提取 unigram + bigram 作为特征（即单个token和token的两两组合）
   - 构建 BM25 倒排索引（`postings`）及样本长度信息  
   - 同时建立直接匹配表（规范化文本 → 标签） 即：要是测试集中有和训练集中一模一样的样本则直接匹配

2. **预测阶段（`predict`）**  
   - **直接匹配捷径**：若测试文本在训练集中完全一致，直接返回标签（零 API 调用）  
   - **BM25 检索**：对测试文本计算与所有训练样本的 BM25 得分，取 top-K（控制各标签配额）  
   - **构造 Few-shot Prompt**：系统消息固定角色，用户消息包含检索到的示例（格式：`Post: ... Label: ...`）  
   - **调用 LLM**：强制要求输出 `<reasoning>...</reasoning>` 和 `<label>0/1</label>`  
   - **解析与回退**：正则提取标签和解释；解析失败则回退到 BM25 最高分样本的标签

**优势**：  
- 利用 BM25 捕捉事件关键词（含 `#`、`@`、网址）  
- Few-shot 引导 LLM 模仿标签模式，同时生成解释  
- 若直接匹配可节省 API 调用，提高效率

## 4. 环境配置

安装依赖：
```bash
pip install -r requirements.txt
```
运行测试脚本：
```bash
python run.py
```
**注意**：`llm_client.py` 中已配置好学校 API 密钥，无需修改，除非达到了限额需要换一个人重新申请

## 5. v1.0 的运行结果

### 评估结果

Accuracy:  0.7930
Precision: 0.7840
Recall:    0.7257
F1 Score:  0.7537

### 预测示例（前5条）

样本 1:
文本: So, to sum up: 1) Darren Wilson KNEW NOTHING of the robbery, 2) shot #MikeBrown over jaywalking, and 3) was allowed to escape #Ferguson.
真实标签: 1
预测标签: 1
解释: The post draws a conclusion that Darren Wilson knew nothing of the robbery and shot Mike Brown over jaywalking, implying a controversial interpretation that aligns with rumor-like narrative. The train...

样本 2:
文本: BREAKING: #Ferguson police chief just announced that officer Darren Wilson shot the unarmed teen, Michael Brown.
真实标签: 1
预测标签: 1
解释: The post reports a breaking news event about the Ferguson police chief announcing that officer Darren Wilson shot Michael Brown. While this is a similar statement to several labeled posts, the trainin...

样本 3:
文本: so ... they clearly released that video  only to shame &amp; blame the victim. #Ferguson #MikeBrown
真实标签: 1
预测标签: 1
解释: The post claims that a video was released specifically to shame and blame the victim, which is an accusation of malicious intent and a conspiracy-like interpretation. This matches the pattern of the r...

样本 4:
文本: BREAKING: #Anonymous has obtained audio files of police dispatch and EMS during the #MikeBrown shooting. Will release ASAP. #Ferguson
真实标签: 1
预测标签: 0
解释: The post claims that Anonymous has obtained audio files of police dispatch and EMS during the Mike Brown shooting and will release them ASAP. This is similar to some of the labeled examples. In the tr...

样本 5:
文本: #Ferguson police are embarking on what can only be described as an elaborate smear campaign of Michael Brown http://t.co/SaLZExqR1D
真实标签: 1
预测标签: 1
解释: The post makes a claim that Ferguson police are conducting an elaborate smear campaign against Michael Brown. This is an unverified, accusatory statement that goes beyond reporting facts. In the train...

## 6. 后续改进方向建议
要是只想小改的话，一些可能的建议：
- 优化 Prompt，引导思考等（目前的提示词可能比较初级）
- 换更好的模型：现在是 DeepSeek V3.2（常规模式），还要思考模式和其他模型，详见 https://claw.sjtu.edu.cn/guide/sjtu-api/
- 优化 BM25 参数（k1, b）或改用其他检索方式（如向量检索）
- 尝试不同 Few-shot 示例选取策略（如按事件平衡）
- 由于LLM的输出结果具有一定的不稳定性，可以考虑对每一个样本重复预测3-5次取最多的lable，但是运行时间会比较长，且可能对正确率影响不大

要是有更好的策略的话，做大改也是很好的（现在的思路偏向于 hardness 工程，只做了检索+一次LLM调用）

## 7. 其他注意事项
- 学校的API有限速，一分钟最多调用十次，完整遍历完一次验证集需要40分钟以上，好慢好慢（要是觉得太慢可以用 Qwen 的免费额度，改 llm_client.py 即可）
