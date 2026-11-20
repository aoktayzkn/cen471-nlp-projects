
## prompt=hardened, T=0.0, seed=42

                             prompt_only   schema
n                               200.0000 200.0000
raw_json_rate                     0.9950   1.0000
cleaned_json_rate                 0.9950   1.0000
schema_valid_rate                 0.9700   1.0000
accuracy                          0.9350   0.9600
macro_f1                          0.9492   0.9600
accuracy_sentiment_salvaged       0.9600   0.9600
n_schema_valid                  194.0000 200.0000
accuracy_on_schema_valid          0.9639   0.9600
truncated_outputs                 0.0000   0.0000
mean_latency_ms                 683.7206 705.2102
entity_n                        100.0000 100.0000
ent_strict_P                      0.6068   0.5977
ent_strict_R                      0.7313   0.7649
ent_strict_F1                     0.6633   0.6710
ent_strict_F1_salvaged            0.6700   0.6710
ent_lenient_P                     0.6563   0.6472
ent_lenient_R                     0.7910   0.8284
ent_lenient_F1                    0.7174   0.7267
ent_lenient_F1_salvaged           0.7236   0.7267

Paired comparison: {"paired_n": 200, "identical_raw_output": 194, "identical_sentiment": 194, "correct_only_prompt_only": 0, "correct_only_schema": 5, "mcnemar_p": 0.0625}

## prompt=hardened, T=1.0, seed=42

                             prompt_only   schema
n                               200.0000 200.0000
raw_json_rate                     0.9950   1.0000
cleaned_json_rate                 0.9950   1.0000
schema_valid_rate                 0.9750   1.0000
accuracy                          0.9350   0.9550
macro_f1                          0.9468   0.9550
accuracy_sentiment_salvaged       0.9550   0.9550
n_schema_valid                  195.0000 200.0000
accuracy_on_schema_valid          0.9590   0.9550
truncated_outputs                 0.0000   0.0000
mean_latency_ms                 696.0130 742.1802
entity_n                        100.0000 100.0000
ent_strict_P                      0.5970   0.5903
ent_strict_R                      0.7351   0.7687
ent_strict_F1                     0.6589   0.6677
ent_strict_F1_salvaged            0.6766   0.6677
ent_lenient_P                     0.6485   0.6418
ent_lenient_R                     0.7985   0.8358
ent_lenient_F1                    0.7157   0.7261
ent_lenient_F1_salvaged           0.7327   0.7261

Paired comparison: {"paired_n": 200, "identical_raw_output": 195, "identical_sentiment": 195, "correct_only_prompt_only": 0, "correct_only_schema": 4, "mcnemar_p": 0.125}

## prompt=minimal, T=0.0, seed=42

                             prompt_only   schema
n                               200.0000 200.0000
raw_json_rate                     0.6400   1.0000
cleaned_json_rate                 0.9950   1.0000
schema_valid_rate                 0.9800   1.0000
accuracy                          0.9450   0.9600
macro_f1                          0.9545   0.9600
accuracy_sentiment_salvaged       0.9600   0.9600
n_schema_valid                  196.0000 200.0000
accuracy_on_schema_valid          0.9643   0.9600
truncated_outputs                 0.0000   0.0000
mean_latency_ms                 717.0634 747.9243
entity_n                        100.0000 100.0000
ent_strict_P                      0.5932   0.5857
ent_strict_R                      0.7836   0.7649
ent_strict_F1                     0.6752   0.6634
ent_strict_F1_salvaged            0.6752   0.6634
ent_lenient_P                     0.6384   0.6371
ent_lenient_R                     0.8433   0.8321
ent_lenient_F1                    0.7267   0.7217
ent_lenient_F1_salvaged           0.7267   0.7217

Paired comparison: {"paired_n": 200, "identical_raw_output": 126, "identical_sentiment": 196, "correct_only_prompt_only": 0, "correct_only_schema": 3, "mcnemar_p": 0.25}

## prompt=minimal, T=1.0, seed=42

                             prompt_only   schema
n                               200.0000 200.0000
raw_json_rate                     0.5400   1.0000
cleaned_json_rate                 0.9950   1.0000
schema_valid_rate                 0.9750   1.0000
accuracy                          0.9500   0.9700
macro_f1                          0.9620   0.9700
accuracy_sentiment_salvaged       0.9700   0.9700
n_schema_valid                  195.0000 200.0000
accuracy_on_schema_valid          0.9744   0.9700
truncated_outputs                 0.0000   0.0000
mean_latency_ms                 782.9152 793.2419
entity_n                        100.0000 100.0000
ent_strict_P                      0.6178   0.6006
ent_strict_R                      0.8022   0.8134
ent_strict_F1                     0.6981   0.6910
ent_strict_F1_salvaged            0.7011   0.6910
ent_lenient_P                     0.6408   0.6253
ent_lenient_R                     0.8321   0.8470
ent_lenient_F1                    0.7240   0.7195
ent_lenient_F1_salvaged           0.7270   0.7195

Paired comparison: {"paired_n": 200, "identical_raw_output": 106, "identical_sentiment": 195, "correct_only_prompt_only": 0, "correct_only_schema": 4, "mcnemar_p": 0.125}
