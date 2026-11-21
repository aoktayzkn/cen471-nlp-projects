# Classical baselines (sentiment only, 200 reviews)

## 5-fold CV (160 training reviews per fold)

| model | mean accuracy | per fold |
|---|---|---|
| TF-IDF + LR | 0.930 | 0.925, 0.900, 1.000, 0.950, 0.875 |
| BoW counts + LR | 0.865 | 0.900, 0.875, 0.950, 0.850, 0.750 |

## Only 10 labelled examples (5+5), test on the other 190, 30 random draws

| model | mean accuracy | min | max |
|---|---|---|---|
| TF-IDF + LR | 0.617 | 0.537 | 0.705 |
| BoW counts + LR | 0.570 | 0.489 | 0.674 |

## Most influential words (fit on all 200 reviews)

- TF-IDF + LR: negative = ['horror', 'bad', 'just', 'old', 'to', 'but', 'damme', 'action']; positive = ['bugs', 'life', 'of', 'andrews', 'great', 'story', 'nanny', 'julie']
- BoW counts + LR: negative = ['who', 'bad', 'out', 'horror', 'old', 'plot', 'but', 'just']; positive = ['indian', 'will', 'nanny', 'bugs', 'julie', 'great', 'story', 'life']
