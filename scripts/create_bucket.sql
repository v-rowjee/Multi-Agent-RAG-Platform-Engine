-- Private object storage for DataFrame-backed Parquet datasets.
-- Run once in the Supabase SQL Editor.

begin;

insert into storage.buckets (
    id,
    name,
    public,
    file_size_limit,
    allowed_mime_types
)
values (
    'datasets',
    'datasets',
    false,
    null, -- no bucket-specific file size limit
    array[
        'application/vnd.apache.parquet',
        'application/octet-stream'
    ]::text[]
)
on conflict (id) do update
set
    name = excluded.name,
    public = excluded.public,
    file_size_limit = null,
    allowed_mime_types = excluded.allowed_mime_types;

-- FastAPI accesses this private bucket using the service-role key.
-- No public storage policies are required.

commit;