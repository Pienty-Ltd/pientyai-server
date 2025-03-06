-- Drop existing functions
DROP FUNCTION IF EXISTS public.refresh_dashboard_stats_views() CASCADE;

-- Create refresh function
CREATE OR REPLACE FUNCTION public.refresh_dashboard_stats_views()
RETURNS void AS $BODY$
BEGIN
    REFRESH MATERIALIZED VIEW mv_user_stats;
    REFRESH MATERIALIZED VIEW mv_organization_stats;
END;
$BODY$ LANGUAGE plpgsql;
