-- Verify figure image_url population after running scripts/upload_figures.py.
-- Run in the Supabase SQL Editor (or psql against SUPABASE_DB_URL).

-- 1) How many figures now have a public URL? Expect have_url = total = 63.
select count(*) filter (where image_url is not null) as have_url,
       count(*)                                      as total
from figures;

-- 2) Any figure still missing a URL? (Expect 0 rows.)
select figure_id, image_path
from figures
where image_url is null
order by figure_id;

-- 3) Spot-check a few resolved URLs to open in a browser.
select figure_id, image_url
from figures
order by figure_id
limit 5;
