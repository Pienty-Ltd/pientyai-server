-- Drop existing function if exists
DROP FUNCTION IF EXISTS public.refresh_dashboard_stats_views();

-- Create refresh function
CREATE OR REPLACE FUNCTION public.refresh_dashboard_stats_views()
    RETURNS void
    LANGUAGE plpgsql
    SECURITY DEFINER
AS
$$
BEGIN
    REFRESH MATERIALIZED VIEW mv_user_stats;
    REFRESH MATERIALIZED VIEW mv_organization_stats;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION public.refresh_dashboard_stats_views() TO public;