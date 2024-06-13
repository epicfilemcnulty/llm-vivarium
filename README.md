# LLM Vivarium

Scripts & configuration files to train Mamba models with byte-level tokenizer.
Requires [bltzr](https://pypi.org/project/bltzr/) package, which
provides the byte-level tokenizer and `SqlDataset` class for working
with datasets in a PostgreSQL database.

## Training

The `SqlDataset` class from the `bltzr` package assumes that you have your training data in a PostgreSQL database.
It requires a dataset table as an argument. This table should be defined as follows:

```sql
CREATE TABLE IF NOT EXISTS dataset (
    id BIGSERIAL PRIMARY KEY,
    tbl VARCHAR(30) NOT NULL,
    ref_id INTEGER NOT NULL
);
```

The `tbl` field should be a name of a table, and the `ref_id` should be the id of the row in that table.
The row should have a `content` text field, storing the actual textual/binary content of your dataset. 

You should also have two functions defined in your database, which are used to get dataset items
and their length. For example:

```sql
CREATE OR REPLACE FUNCTION get_dataset_item_len(tbl TEXT, ref_id BIGINT, metadata BOOL)
RETURNS INTEGER
AS $$
    local metadata = metadata or false
    local query = "SELECT octet_length(content) as len FROM " .. tbl .. " WHERE id = " .. ref_id .. ";"
    local res = spi.execute(query)
    local len = res[1].len or 0
    if metadata then
        if tbl:match("^wiki") then
            len = len + 21 -- <META> + {"src":"wikipedia"} + </META>
        end
        if tbl == "rfc" then
            len = len + 15 -- <META> + {"src":"RFC"} + </META>
        end
    end
    return len + 2 -- Two embracing special tokens
$$ LANGUAGE pllua;
 
CREATE OR REPLACE FUNCTION get_dataset_item(tbl TEXT, ref_id BIGINT, metadata BOOL)
RETURNS JSONB 
AS $$
    local metadata = metadata or false
    local query = "SELECT content FROM " .. tbl .. " WHERE id = " .. ref_id .. ";"
    local res = spi.execute(query)
    local content = res[1].content
    local meta
    if metadata then
        local src = "wikipedia"
        if tbl == "rfc" then
            src = "RFC"
        end
        meta = '{"src":"' .. src .. '"}'
    end
    local msgs = {
        { kind = "txt", content = content, meta = meta }
    }
    return msgs
$$ LANGUAGE pllua;
```
