# Multi-Label Classification System for Customer Interaction Tagging  
**Module**: Engineering and Evaluating AI Systems (H9EEAI)  
**Institution**: National College of Ireland (MSc in Artificial Intelligence)

---

### Project Overview

This project is a software engineering and machine learning solution developed as part of the **H9EEAI** module at NCI. The goal is to **design, implement, and evaluate architectures** for a real-world **multi-label classification task**.

Simulating the role of an AI engineering team enhancing a SaaS product that classifies customer messages into multiple interdependent categories:

- **Intent** (e.g., Feedback, Support, Other)  
- **Tone** (e.g., Positive, Neutral, Negative)  
- **Resolution Type** (e.g., Auto-reply, Manual Review, Escalate)

The system transitions from an existing single-label pipeline to a more complex **multi-label architecture** that captures label dependencies.

---

### Architectures Designed

#### 1. Chained Multi-Output Architecture (Implemented)

- Each model’s prediction feeds into the next stage.
- Models are trained sequentially and executed in a pipeline.
  
```
[Text Input] → [Model 1: Intent] → [Model 2: Tone] → [Model 3: Resolution Type]
```

---

###  Evaluation Metrics

- **Label Accuracy** (per class)
- **f1-score, recall & precision** (full sequence correct)


### Submission Details

- **Module**: Engineering and Evaluating AI Systems (H9EEAI)  
- **Lecturer**: Jaswinder Singh  
- **Assessment**: Continuous Assessment 
- **Institution**: National College of Ireland  

---

### Team Members

- Danilo Angel Tito Rodriguez
- Jose Manuel Lopez Garcia

(Each member contributed to design, implementation, and documentation)

