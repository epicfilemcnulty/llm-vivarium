# LLM Vivarium

Scripts & configuration files to train Mamba & Mamba2 models with byte-level tokenizer.
Requires [bltzr](https://pypi.org/project/bltzr/) package, which
provides the byte-level tokenizer and `SqlDataset` class for working
with datasets in a PostgreSQL database.

You should also have [mamba-ssm](https://github.com/state-spaces/mamba) package installed.

## Training

The `SqlDataset` class from the `bltzr` package assumes that you have your training data in a PostgreSQL database.
It requires a dataset table as an argument. This table should have the following schema:

```sql
CREATE TABLE IF NOT EXISTS dataset (
    id BIGSERIAL PRIMARY KEY,
    tbl VARCHAR(30) NOT NULL,
    ref_id INTEGER NOT NULL
);
```

The `tbl` field should be a name of a table, and the `ref_id` should be the id of the row in that table.
The row should have a `content` text/binary field, storing the actual textual/binary content of your dataset. 

You should also have two functions defined in your database, `get_dataset_item` and `get_dataset_items_len`,
which are used to get dataset items and their length (duh!). The signature of the functions:

```sql
CREATE OR REPLACE FUNCTION get_dataset_items_len(tbl TEXT[], ref_id BIGINT[], metadata BOOL)
RETURNS INTEGER[]
... 
 
CREATE OR REPLACE FUNCTION get_dataset_item(tbl TEXT, ref_id BIGINT, metadata BOOL)
RETURNS JSONB 
AS $$
    -- Replace with some code to actually fetch the dataset item =)
    local content = "Example content"
    local meta
    if metadata then
        local meta = '{"src":"Source","title":"Title"}'
    end
    -- and then you should return it in the format below:
    local msgs = {
        { kind = "txt", content = content, meta = meta }
    }
    return msgs
$$ LANGUAGE pllua;
```
