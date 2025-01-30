# MoFO
Code for MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning

# How to use MoFO in your code

Install torch (>=1.8.0) and run the following commands
```python
import algorithms_MoFO

optimizer = algorithms.AdamW_MoFO(
        model = model, lr=args.learning_rate, weight_decay=args.weight_decay,fraction=args.MoFO_fraction)
```

Hyperparameter: `MoFO_fraction` is the parameter update fraction of weight in each iteration. For example, `MoFO_fraction=0.15` means 15% parameter with the highest momentum will be updated in each iteration. We recommend setting `MoFO_fraction` between 5% and 20%.


 
