-- Function to update dashboard statistics
CREATE OR REPLACE FUNCTION update_dashboard_stats()
RETURNS void AS $$
BEGIN
    -- Update user statistics
    INSERT INTO dashboard_stats (
        user_id,
        total_knowledge_base_count,
        total_file_count,
        total_storage_used,
        last_activity_date
    )
    SELECT 
        u.id as user_id,
        COUNT(DISTINCT kb.id) as total_knowledge_base_count,
        COUNT(DISTINCT f.id) as total_file_count,
        COALESCE(SUM(f.file_size), 0) as total_storage_used,
        MAX(GREATEST(kb.updated_at, f.updated_at)) as last_activity_date
    FROM users u
    LEFT JOIN files f ON f.user_id = u.id
    LEFT JOIN knowledge_base kb ON kb.file_id = f.id
    GROUP BY u.id
    ON CONFLICT (user_id) WHERE user_id IS NOT NULL
    DO UPDATE SET
        total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
        total_file_count = EXCLUDED.total_file_count,
        total_storage_used = EXCLUDED.total_storage_used,
        last_activity_date = EXCLUDED.last_activity_date,
        last_updated = CURRENT_TIMESTAMP;

    -- Update organization statistics
    INSERT INTO dashboard_stats (
        organization_id,
        total_knowledge_base_count,
        total_file_count,
        total_storage_used,
        last_activity_date
    )
    SELECT 
        o.id as organization_id,
        COUNT(DISTINCT kb.id) as total_knowledge_base_count,
        COUNT(DISTINCT f.id) as total_file_count,
        COALESCE(SUM(f.file_size), 0) as total_storage_used,
        MAX(GREATEST(kb.updated_at, f.updated_at)) as last_activity_date
    FROM organizations o
    LEFT JOIN files f ON f.organization_id = o.id
    LEFT JOIN knowledge_base kb ON kb.file_id = f.id
    GROUP BY o.id
    ON CONFLICT (organization_id) WHERE organization_id IS NOT NULL
    DO UPDATE SET
        total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
        total_file_count = EXCLUDED.total_file_count,
        total_storage_used = EXCLUDED.total_storage_used,
        last_activity_date = EXCLUDED.last_activity_date,
        last_updated = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create a cron job to run the function every hour
SELECT cron.schedule('update_dashboard_stats_hourly', '0 * * * *', 'SELECT update_dashboard_stats()');
