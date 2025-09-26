const express = require('express');
const app = express();
const pg = require('pg');
const { Pool } = pg;

const pool = new Pool({
        "user": "postgres",
        "host": "postgis",
        "database": "canyon",
        "password": "1234",
        "port": 5432
});

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/', express.static('www'));

app.get('/usc/api/points', (req, res) => {
        //create geojson data from class_and_pollution table and select id column only
        const sql = `SELECT jsonb_build_object(
                'type', 'FeatureCollection',
                'features', jsonb_agg(features.feature)
        ) AS geojson
        FROM (
                SELECT jsonb_build_object(
                        'type', 'Feature',
                        'id', id,
                        'geometry', ST_AsGeoJSON(geom)::jsonb,
                        'properties', to_jsonb(inputs) -- 'geom'
                ) AS feature
                FROM (SELECT id, lat, lon, label, conf, azimuth, geom FROM t_class_and_pollution) inputs
        ) features;`;
        pool.query(sql, (err, result) => {
                if (err) {
                        console.error(err);
                        res.status(500).json({ error: 'Database query error' });
                } else {
                        res.json(result.rows[0].geojson);
                }
        });
});

app.get('/usc/api/points/:id', (req, res) => {
        const id = req.params.id;
        const sql = `SELECT * FROM t_class_and_pollution WHERE id = $1`;
        pool.query(sql, [id], (err, result) => {
                if (err) {
                        console.error(err);
                        res.status(500).json({ error: 'Database query error' });
                } else {
                        if (result.rows.length === 0) {
                                res.status(404).json({ error: 'Point not found' });
                        } else {
                                res.json(result.rows[0]);
                        }
                }
        });
});

app.get('/usc/api/point_chart/:id', (req, res) => {
        const id = req.params.id;
        const sql = `SELECT * FROM class_and_pollution WHERE id = $1`;
        pool.query(sql, [id], (err, result) => {
                if (err) {
                        console.error(err);
                        res.status(500).json({ error: 'Database query error' });
                } else {
                        if (result.rows.length === 0) {
                                res.status(404).json({ error: 'Point not found' });
                        } else {
                                res.json(result.rows[0]);
                        }
                }
        });
});

app.get('/usc/api/sathonpoints', (req, res) => {
        //create geojson data from class_and_pollution table and select id column only
        const sql = `SELECT jsonb_build_object(
                'type', 'FeatureCollection',
                'features', jsonb_agg(features.feature)
        ) AS geojson
        FROM (
                SELECT jsonb_build_object(
                        'type', 'Feature',
                        'id', id,
                        'geometry', ST_AsGeoJSON(geom)::jsonb,
                        'properties', to_jsonb(inputs) -- 'geom'
                ) AS feature
                FROM (SELECT id, lat, lon,azimuth, geom FROM sathonpoint_azimuth_4326) inputs
        ) features;`;
        pool.query(sql, (err, result) => {
                if (err) {
                        console.error(err);
                        res.status(500).json({ error: 'Database query error' });
                } else {
                        res.json(result.rows[0].geojson);
                }
        });
});

app.get('/usc/api/sathonpoints/:id', (req, res) => {
        const id = req.params.id;
        const sql = `SELECT * FROM sathonpoint_azimuth_4326 WHERE id = $1`;
        pool.query(sql, [id], (err, result) => {
                if (err) {
                        console.error(err);
                        res.status(500).json({ error: 'Database query error' });
                } else {
                        if (result.rows.length === 0) {
                                res.status(404).json({ error: 'Point not found' });
                        } else {
                                res.json(result.rows[0]);
                        }
                }
        });
});


const PORT = 3000;
app.listen(PORT, () => {
        console.log(`Server running at http://localhost:${PORT}`);
}); 