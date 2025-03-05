-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_files_user_id ON files(user_id);
CREATE INDEX IF NOT EXISTS idx_files_organization_id ON files(organization_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_file_id ON knowledge_base(file_id);
CREATE INDEX IF NOT EXISTS idx_user_organizations_user_id ON user_organizations(user_id);
CREATE INDEX IF NOT EXISTS idx_user_organizations_organization_id ON user_organizations(organization_id);

-- Create materialized view for user statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_stats AS
SELECT 
    u.id as user_id,
    COUNT(DISTINCT kb.id) as total_knowledge_base_count,
    COUNT(DISTINCT f.id) as total_file_count,
    COALESCE(SUM(f.file_size), 0) as total_storage_used,
    MAX(GREATEST(kb.updated_at, f.updated_at)) as last_activity_date
FROM users u
LEFT JOIN files f ON f.user_id = u.id
LEFT JOIN knowledge_base kb ON kb.file_id = f.id
GROUP BY u.id;

-- Create materialized view for organization statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_organization_stats AS
SELECT 
    o.id as organization_id,
    COUNT(DISTINCT kb.id) as total_knowledge_base_count,
    COUNT(DISTINCT f.id) as total_file_count,
    COALESCE(SUM(f.file_size), 0) as total_storage_used,
    MAX(GREATEST(kb.updated_at, f.updated_at)) as last_activity_date
FROM organizations o
LEFT JOIN files f ON f.organization_id = o.id
LEFT JOIN knowledge_base kb ON kb.file_id = f.id
GROUP BY o.id;

-- Function to refresh materialized views concurrently
CREATE OR REPLACE FUNCTION refresh_dashboard_stats_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_organization_stats;
END;
$$ LANGUAGE plpgsql;

-- Function to update dashboard statistics using materialized views
CREATE OR REPLACE FUNCTION update_dashboard_stats(batch_size integer DEFAULT 1000)
RETURNS void AS $$
DECLARE
    user_cursor CURSOR FOR 
        SELECT user_id, total_knowledge_base_count, total_file_count, 
               total_storage_used, last_activity_date 
        FROM mv_user_stats;
    org_cursor CURSOR FOR 
        SELECT organization_id, total_knowledge_base_count, total_file_count, 
               total_storage_used, last_activity_date 
        FROM mv_organization_stats;
    batch_counter integer := 0;
BEGIN
    -- Process user statistics in batches
    FOR user_stat IN user_cursor LOOP
        INSERT INTO dashboard_stats (
            user_id,
            total_knowledge_base_count,
            total_file_count,
            total_storage_used,
            last_activity_date
        ) VALUES (
            user_stat.user_id,
            user_stat.total_knowledge_base_count,
            user_stat.total_file_count,
            user_stat.total_storage_used,
            user_stat.last_activity_date
        )
        ON CONFLICT (user_id) WHERE user_id IS NOT NULL
        DO UPDATE SET
            total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
            total_file_count = EXCLUDED.total_file_count,
            total_storage_used = EXCLUDED.total_storage_used,
            last_activity_date = EXCLUDED.last_activity_date,
            last_updated = CURRENT_TIMESTAMP;

        batch_counter := batch_counter + 1;
        IF batch_counter >= batch_size THEN
            COMMIT;
            batch_counter := 0;
        END IF;
    END LOOP;

    -- Process organization statistics in batches
    batch_counter := 0;
    FOR org_stat IN org_cursor LOOP
        INSERT INTO dashboard_stats (
            organization_id,
            total_knowledge_base_count,
            total_file_count,
            total_storage_used,
            last_activity_date
        ) VALUES (
            org_stat.organization_id,
            org_stat.total_knowledge_base_count,
            org_stat.total_file_count,
            org_stat.total_storage_used,
            org_stat.last_activity_date
        )
        ON CONFLICT (organization_id) WHERE organization_id IS NOT NULL
        DO UPDATE SET
            total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
            total_file_count = EXCLUDED.total_file_count,
            total_storage_used = EXCLUDED.total_storage_used,
            last_activity_date = EXCLUDED.last_activity_date,
            last_updated = CURRENT_TIMESTAMP;

        batch_counter := batch_counter + 1;
        IF batch_counter >= batch_size THEN
            COMMIT;
            batch_counter := 0;
        END IF;
    END LOOP;

    -- Final commit for any remaining records
    IF batch_counter > 0 THEN
        COMMIT;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to update specific user's statistics
CREATE OR REPLACE FUNCTION update_user_stats(p_user_id integer)
RETURNS void AS $$
BEGIN
    WITH user_stats AS (
        SELECT 
            u.id as user_id,
            COUNT(DISTINCT kb.id) as total_knowledge_base_count,
            COUNT(DISTINCT f.id) as total_file_count,
            COALESCE(SUM(f.file_size), 0) as total_storage_used,
            MAX(GREATEST(kb.updated_at, f.updated_at)) as last_activity_date
        FROM users u
        LEFT JOIN files f ON f.user_id = u.id
        LEFT JOIN knowledge_base kb ON kb.file_id = f.id
        WHERE u.id = p_user_id
        GROUP BY u.id
    )
    INSERT INTO dashboard_stats (
        user_id,
        total_knowledge_base_count,
        total_file_count,
        total_storage_used,
        last_activity_date
    )
    SELECT * FROM user_stats
    ON CONFLICT (user_id) WHERE user_id IS NOT NULL
    DO UPDATE SET
        total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
        total_file_count = EXCLUDED.total_file_count,
        total_storage_used = EXCLUDED.total_storage_used,
        last_activity_date = EXCLUDED.last_activity_date,
        last_updated = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Function to update specific organization's statistics
CREATE OR REPLACE FUNCTION update_organization_stats(p_organization_id integer)
RETURNS void AS $$
BEGIN
    WITH org_stats AS (
        SELECT 
            o.id as organization_id,
            COUNT(DISTINCT kb.id) as total_knowledge_base_count,
            COUNT(DISTINCT f.id) as total_file_count,
            COALESCE(SUM(f.file_size), 0) as total_storage_used,
            MAX(GREATEST(kb.updated_at, f.updated_at)) as last_activity_date
        FROM organizations o
        LEFT JOIN files f ON f.organization_id = o.id
        LEFT JOIN knowledge_base kb ON kb.file_id = f.id
        WHERE o.id = p_organization_id
        GROUP BY o.id
    )
    INSERT INTO dashboard_stats (
        organization_id,
        total_knowledge_base_count,
        total_file_count,
        total_storage_used,
        last_activity_date
    )
    SELECT * FROM org_stats
    ON CONFLICT (organization_id) WHERE organization_id IS NOT NULL
    DO UPDATE SET
        total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
        total_file_count = EXCLUDED.total_file_count,
        total_storage_used = EXCLUDED.total_storage_used,
        last_activity_date = EXCLUDED.last_activity_date,
        last_updated = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create cron jobs
SELECT cron.schedule('refresh_views_hourly', '30 * * * *', 
    $$SELECT refresh_dashboard_stats_views()$$);
SELECT cron.schedule('update_stats_hourly', '0 * * * *', 
    $$SELECT update_dashboard_stats(1000)$$);