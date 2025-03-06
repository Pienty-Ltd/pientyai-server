-- Drop existing functions
DROP FUNCTION IF EXISTS public.update_dashboard_stats(integer) CASCADE;
DROP FUNCTION IF EXISTS public.update_user_stats(integer) CASCADE;
DROP FUNCTION IF EXISTS public.update_organization_stats(integer) CASCADE;

-- Create update stats function
CREATE OR REPLACE FUNCTION public.update_dashboard_stats(batch_size integer DEFAULT 1000)
RETURNS void AS $BODY$
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
            batch_counter := 0;
            COMMIT;
        END IF;
    END LOOP;

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
            batch_counter := 0;
            COMMIT;
        END IF;
    END LOOP;
END;
$BODY$ LANGUAGE plpgsql;

-- Create user stats update function
CREATE OR REPLACE FUNCTION public.update_user_stats(p_user_id integer)
RETURNS void AS $BODY$
BEGIN
    WITH user_stats AS (
        SELECT 
            u.id as user_id,
            COUNT(DISTINCT kb.id) as total_knowledge_base_count,
            COUNT(DISTINCT f.id) as total_file_count,
            COALESCE(SUM(f.file_size), 0) as total_storage_used,
            MAX(GREATEST(COALESCE(kb.updated_at, '1970-01-01'), COALESCE(f.updated_at, '1970-01-01'))) as last_activity_date
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
    SELECT 
        user_id,
        total_knowledge_base_count,
        total_file_count,
        total_storage_used,
        last_activity_date
    FROM user_stats
    ON CONFLICT (user_id) WHERE user_id IS NOT NULL
    DO UPDATE SET
        total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
        total_file_count = EXCLUDED.total_file_count,
        total_storage_used = EXCLUDED.total_storage_used,
        last_activity_date = EXCLUDED.last_activity_date,
        last_updated = CURRENT_TIMESTAMP;
END;
$BODY$ LANGUAGE plpgsql;

-- Create organization stats update function
CREATE OR REPLACE FUNCTION public.update_organization_stats(p_organization_id integer)
RETURNS void AS $BODY$
BEGIN
    WITH org_stats AS (
        SELECT 
            o.id as organization_id,
            COUNT(DISTINCT kb.id) as total_knowledge_base_count,
            COUNT(DISTINCT f.id) as total_file_count,
            COALESCE(SUM(f.file_size), 0) as total_storage_used,
            MAX(GREATEST(COALESCE(kb.updated_at, '1970-01-01'), COALESCE(f.updated_at, '1970-01-01'))) as last_activity_date
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
    SELECT 
        organization_id,
        total_knowledge_base_count,
        total_file_count,
        total_storage_used,
        last_activity_date
    FROM org_stats
    ON CONFLICT (organization_id) WHERE organization_id IS NOT NULL
    DO UPDATE SET
        total_knowledge_base_count = EXCLUDED.total_knowledge_base_count,
        total_file_count = EXCLUDED.total_file_count,
        total_storage_used = EXCLUDED.total_storage_used,
        last_activity_date = EXCLUDED.last_activity_date,
        last_updated = CURRENT_TIMESTAMP;
END;
$BODY$ LANGUAGE plpgsql;
