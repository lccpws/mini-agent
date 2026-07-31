# Mini Agent 项目介绍

## 概述

Mini Agent 是一个轻量级的 AI Agent 框架，支持 ReAct 和 Plan+ReAct 两种推理模式。

## 核心功能

### 1. ReAct 模式
ReAct（Reasoning + Acting）是一种让 AI 模型交替进行推理和行动的框架。模型会：
- 思考当前情况
- 决定下一步行动
- 执行工具调用
- 观察结果
- 继续推理

### 2. Plan+ReAct 模式
先制定执行计划，然后按计划逐步执行。适用于复杂任务。

### 3. 记忆系统
- 工作记忆：当前任务的临时记忆
- 短期记忆：对话历史
- 长期记忆：持久化的重要信息

### 4. 工具系统
支持自定义工具扩展，包括：
- 数学计算
- 天气查询
- 文本处理

## 技术架构

- Python 3.10+
- OpenAI API
- 向量存储（用于记忆和知识库）

## 使用示例

```python
from mini_agent import ReactAgent

agent = ReactAgent(controller)
result = agent.run("北京天气怎么样？")
```