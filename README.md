# RTAME: 基于关系类型的半动态知识图谱模型集成

本项目实现了一个名为 **RTAME (Relation-Type Aware Model Ensembling)** 的知识图谱补全框架。其核心思想是，根据关系的内在特性（如出现频率），动态地、有针对性地集成**基于结构的模型 (CompoundE)** 和**基于文本的模型 (SimKGC)** 的预测结果，旨在实现比单一模型或简单集成更优越、且比复杂集成模型更稳健和可复现的性能。

本项目已被参数化，以支持 `FB15k-237` 和 `WN18RR` 两个数据集。

## 项目结构

```
KGC-E/
├── CompoundE/              # 包含 CompoundE 模型及其修改后的集成评估代码
├── SimKGC/                 # 包含 SimKGC 模型及生成其预测分数的代码
├── analyze_relation_frequency.py # 用于分析关系频率的通用脚本
├── relation_types.json     # FB15k-237 的关系分类文件
├── relation_types_wn18rr.json # WN18RR 的关系分类文件
└── README.md               # 本说明文档
```

## 实验复现流程

请严格按照以下步骤操作，以复现本研究的实验结果。`[DATASET_NAME]` 可以是 `FB15k237` 或 `WN18RR`。

### 第零步：环境配置

在开始前，请确保您已分别配置好 `CompoundE` 和 `SimKGC` 的 Python 环境，并安装了它们各自在 `requirements.txt` 中声明的依赖库。

### 第一步：训练基础模型

在进行模型集成之前，我们需要分别为**每个数据集**获取两个基础模型的预训练检查点 (checkpoint)。

1.  **训练 CompoundE**: 为 `FB15k237` 和 `WN18RR` 分别训练一个 `CompoundE` 模型，并妥善保管好各自的检查点文件。

2.  **训练 SimKGC**: 同样地，为 `FB15k237` 和 `WN18RR` 分别训练一个 `SimKGC` 模型，并妥善保管好各自的检查点文件。

### 第二步：为指定数据集生成 SimKGC 预测分数

此步骤的目的是让 `SimKGC` 对指定数据集的测试集进行预测，并将分数保存下来。

1.  **进入目录**: `cd SimKGC`
2.  **运行脚本**: 执行以下命令。请将 `[DATASET_NAME]` 替换为目标数据集名称，并将 `[CHECKPOINT_PATH]` 替换为**对应数据集的 SimKGC 检查点路径**。

    ```bash
    python generate_simkgc_scores.py --dataset [DATASET_NAME] --checkpoint [CHECKPOINT_PATH]
    ```

    *   **FB15k-237 示例**:
        ```bash
        python generate_simkgc_scores.py --dataset FB15k237 --checkpoint checkpoint/FB15k237/SimKGC_FB15k237_bert-base-uncased/checkpoint_best.pt
        ```

    *   **WN18RR 示例**:
        ```bash
        python generate_simkgc_scores.py --dataset WN18RR --checkpoint path/to/your/wn18rr/simkgc_checkpoint.pt
        ```

3.  **检查产出**: 脚本执行完毕后，会在 `SimKGC/predictions/[DATASET_NAME]/` 目录下生成一个名为 `simkgc_scores.pt` 的文件。

### 第三步：执行 RTAME 集成评估

现在，我们可以为指定的数据集运行最终的集成了。

1.  **进入目录**: `cd CompoundE`
2.  **运行评估**: 执行以下命令。请将 `[DATASET_NAME]` 替换为目标数据集名称，并将 `[CHECKPOINT_DIR]` 替换为**对应数据集的 CompoundE 检查点所在的目录**。

    ```bash
    python run.py --dataset [DATASET_NAME] --do_test --cuda --init_checkpoint [CHECKPOINT_DIR]
    ```

    *   **FB15k-237 示例**:
        ```bash
        python run.py --dataset FB15k237 --do_test --cuda --init_checkpoint log/FB15k237/CompoundE/...
        ```

    *   **WN18RR 示例**:
        ```bash
        python run.py --dataset WN18RR --do_test --cuda --init_checkpoint path/to/your/wn18rr/compounde_checkpoint_dir
        ```

### 第四步：查看结果

脚本执行完毕后，您将在控制台的输出中看到最终的评估结果，格式如下：

```
{'mrr': XXX, 'hits@1': XXX, 'hits@3': XXX, 'hits@10': XXX}
```

这即是 `RTAME` 集成模型在指定数据集上的性能表现。

## 超参数调整

如果您希望进一步优化性能，可以调整以下超参数：

*   **关系划分**: 修改 `relation_types.json` (FB15k237) 或 `relation_types_wn18rr.json` (WN18RR) 文件来调整高/低频关系的划分。
*   **集成权重**: 在 `CompoundE/model.py` 的 `test_step` 函数中，直接修改 `alpha` 和 `beta` 的值。