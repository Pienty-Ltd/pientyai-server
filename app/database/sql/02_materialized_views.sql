-- Drop existing materialized views
DROP MATERIALIZED VIEW IF EXISTS mv_user_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_organization_stats CASCADE;

-- Create materialized view for user statistics
CREATE MATERIALIZED VIEW mv_user_stats AS
SELECT 
    u.id as user_id,
    COUNT(DISTINCT kb.id) as total_knowledge_base_count,
    COUNT(DISTINCT f.id) as total_file_count,
    COALESCE(SUM(f.file_size), 0) as total_storage_used,
    MAX(GREATEST(COALESCE(kb.updated_at, '1970-01-01'), COALESCE(f.updated_at, '1970-01-01'))) as last_activity_date
FROM users u
LEFT JOIN files f ON f.user_id = u.id
LEFT JOIN knowledge_base kb ON kb.file_id = f.id
GROUP BY u.id;

CREATE UNIQUE INDEX idx_mv_user_stats_user_id ON mv_user_stats(user_id);

-- Create materialized view for organization statistics
CREATE MATERIALIZED VIEW mv_organization_stats AS
SELECT 
    o.id as organization_id,
    COUNT(DISTINCT kb.id) as total_knowledge_base_count,
    COUNT(DISTINCT f.id) as total_file_count,
    COALESCE(SUM(f.file_size), 0) as total_storage_used,
    MAX(GREATEST(COALESCE(kb.updated_at, '1970-01-01'), COALESCE(f.updated_at, '1970-01-01'))) as last_activity_date
FROM organizations o
LEFT JOIN files f ON f.organization_id = o.id
LEFT JOIN knowledge_base kb ON kb.file_id = f.id
GROUP BY o.id;

CREATE UNIQUE INDEX idx_mv_organization_stats_organization_id ON mv_organization_stats(organization_id);
