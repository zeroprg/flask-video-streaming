CREATE TABLE objects(hashcode TEXT PRIMARY KEY, currentdate DATE, currentime TIME, type TEXT, frame BLOB, x_dim INTEGER, y_dim INTEGER, cam INTEGER, lastime TIME);
CREATE TABLE statistic(type TEXT, currentime TIME, y INTEGER, text TEXT, cam INTEGER, lastime TIME, hashcodes TEXT);
CREATE INDEX index_currentime ON objects(currentime);
CREATE INDEX index_currentime_stat ON statistic(currentime);
CREATE INDEX index_lastime_stat ON statistic(lastime);
CREATE INDEX index_lastime ON objects(lastime);
