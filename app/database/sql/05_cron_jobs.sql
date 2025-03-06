-- Drop existing function
DROP FUNCTION IF EXISTS public.manage_dashboard_stats_cron_jobs() CASCADE;

-- Create cron job management function
CREATE OR REPLACE FUNCTION public.manage_dashboard_stats_cron_jobs()
RETURNS void AS $BODY$
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM pg_extension 
        WHERE extname = 'pg_cron'
    ) THEN
        -- Delete existing jobs if they exist
        DELETE FROM cron.job WHERE jobname IN ('refresh_views_hourly', 'update_stats_hourly');

        -- Add new jobs
        INSERT INTO cron.job (jobname, schedule, command, nodename, nodeport, database, username)
        VALUES 
            ('refresh_views_hourly', '*/30 * * * *', 'SELECT refresh_dashboard_stats_views()', 
            'localhost', 5432, current_database(), current_user),
            ('update_stats_hourly', '0 * * * *', 'SELECT update_dashboard_stats(1000)', 
            'localhost', 5432, current_database(), current_user);

        RAISE NOTICE 'Dashboard stats cron jobs have been configured successfully';
    ELSE
        RAISE WARNING 'pg_cron extension is not available. Cron jobs were not scheduled.';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'Error managing cron jobs: %', SQLERRM;
END;
$BODY$ LANGUAGE plpgsql;

-- Initialize cron jobs
SELECT public.manage_dashboard_stats_cron_jobs();
