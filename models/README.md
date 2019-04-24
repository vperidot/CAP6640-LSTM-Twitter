Word-embedding model experiments:

| NAME             | Layers  | Channels/layer  | Dropout |
|------------------|---------|-----------------|---------|
| **words-depth-1**| 1       | 64              | none    |
| words-depth-2    | 2       | 64              | none    |
| words-depth-3    | 3       | 64              | none    | 
| words-dropout-25 | 1       | 64              | 0.25    |
| words-dropout-50 | 1       | 64              | 0.50    |
| words-dropout-65 | 1       | 64              | 0.65    |

Character-embedding model experiments:


| NAME            | Layers  | Channels/layer  | Dropout |
|-----------------|---------|-----------------|---------|
| char-depth-1    | 1       | 64              | none    |
| char-depth-2    | 2       | 64              | none    |
| **char-depth-3**| 3       | 64              | none    | 
| char-dropout-25 | 3       | 64              | 0.25    |
| char-dropout-50 | 3       | 64              | 0.50    |
| char-dropout-65 | 3       | 64              | 0.65    |
