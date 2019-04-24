Word-embedding model experiments:

| NAME             | Embedding | Layers  | Channels/layer  | Dropout |
|------------------|-----------|---------|-----------------|---------|
| **words-depth-1**| word      | 1       | 64              | none    |
| words-depth-2    | word      | 2       | 64              | none    |
| words-depth-3    | word      | 3       | 64              | none    | 
| words-dropout-25 | word      | 1       | 64              | 0.25    |
| words-dropout-50 | word      | 1       | 64              | 0.50    |
| words-dropout-65 | word      | 1       | 64              | 0.65    |

Character-embedding model experiments:

| NAME            | Embedding | Layers  | Channels/layer  | Dropout |
|-----------------|-----------|---------|-----------------|---------|
| char-depth-1    | char      | 1       | 64              | none    |
| char-depth-2    | char      | 2       | 64              | none    |
| **char-depth-3**| char      | 3       | 64              | none    | 
| char-dropout-25 | char      | 3       | 64              | 0.25    |
| char-dropout-50 | char      | 3       | 64              | 0.50    |
| char-dropout-65 | char      | 3       | 64              | 0.65    |

Validation experiments:

| NAME                | Embedding | Layers  | Channels/layer  | Dropout |
|---------------------|-----------|---------|-----------------|---------|
| validation-ellen    | word      | 1       | 64              | none    |
| validation-michelle | word      | 1       | 64              | none    |
