# Mamba Vivarium

Scripts & configs to train Mamba models and run inference. 
Depend on the [mamba_byte_toolkit](https://github.com/epicfilemcnulty/mamba_byte_toolkit) package.

## Training

The dataset class from `mamba_byte_toolkit` assumes that you have your data in a PostgreSQL database.
It requires a dataset table as an argument. This table should be defined as follows:

```sql
CREATE TABLE IF NOT EXISTS dataset (
    id BIGSERIAL PRIMARY KEY,
    tbl VARCHAR(30) NOT NULL,
    ref_id INTEGER NOT NULL
);
```

The `tbl` field should be a name of a table, and the `ref_id` should be the id of the row in that table.
The row should have a `content` text field, storing the actual textual content of your dataset. That's pretty 
much it, with one exception -- if `tbl` field value is `chats`, we assume that this is a special table which must
have fields `len` (integer) and `chat` (JSONB). The chat field should store a chat in the following JSON format:

```JSON
[
   {"kind": "spt", "token": "<CHAT>"}, 
   {"kind": "spt", "token": "<SYS>", "content": "You are an AI assistant"},
   {"kind": "spt", "token": "</SYS>"},
   {"kind": "spt", "token": "<QUERY>", "content": "Hey, how are you?"},
   {"kind": "spt", "token": "</QUERY>"}, 
   {"kind": "spt", "token": "<REPLY>" "content": "Not bad, what about you?"}, 
   {"kind": "spt", "token": "</REPLY>"}, 
   {"kind": "spt", "token": "</CHAT>"}, 
]
```

And the `len` field should have the sum of lengths (in bytes) of every `content` field in the JSON array + 1 byte for every special token.
