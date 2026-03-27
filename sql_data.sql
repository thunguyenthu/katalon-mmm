WITH
spend as (
SELECT
	month_id as month,

	-- ── SPEND (original columns — keep for backward compatibility) ─────────
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%brand%' then cost end) as google_brand_spend,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%competitor%' then cost end) as google_competitor_spend,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%soqr%' then cost end) as google_soqr_annual_report_spend,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%solution%' then cost end) as google_solution_spend,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%retargeting%' then cost end) as google_retargeting_spend,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%abm%' then cost end) as google_abm_spend,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%thought leader%' then cost end) as google_thought_leader_spend,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%ppc%' then cost end) as google_ppc_spend,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) NOT SIMILAR TO '%(ppc|thought leader|abm|retargeting|solution|soqr|competitor|brand)%' then cost end) as google_other_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%brand%' then cost end) as linkedin_brand_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%competitor%' then cost end) as linkedin_competitor_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%soqr%' then cost end) as linkedin_soqr_annual_report_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%solution%' then cost end) as linkedin_solution_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%retargeting%' then cost end) as linkedin_retargeting_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%abm%' then cost end) as linkedin_abm_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%thought leader%' then cost end) as linkedin_thought_leader_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%ppc%' then cost end) as linkedin_ppc_spend,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) NOT SIMILAR TO '%(ppc|thought leader|abm|retargeting|solution|soqr|competitor|brand)%' then cost end) as linkedin_other_spend,
	SUM(CASE WHEN ads_provider = 'Meta' then cost end) as meta_spend,
	SUM(CASE WHEN ads_provider not in ('Google','Linkedin','Meta') then cost end) as other_spend,

	-- ── IMPRESSIONS (awareness / brand channels — use instead of spend in v2) ─
	-- Model grouping: google_brand_spend  = brand + thought_leader
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%brand%' then impressions end) as google_brand_impressions,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%thought leader%' then impressions end) as google_thought_leader_impressions,
	-- Model grouping: google_capture_spend = ppc + competitor + solution + soqr + retargeting + abm
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%competitor%' then impressions end) as google_competitor_impressions,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%soqr%' then impressions end) as google_soqr_impressions,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%solution%' then impressions end) as google_solution_impressions,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%retargeting%' then impressions end) as google_retargeting_impressions,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%abm%' then impressions end) as google_abm_impressions,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%ppc%' then impressions end) as google_ppc_impressions,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) NOT SIMILAR TO '%(ppc|thought leader|abm|retargeting|solution|soqr|competitor|brand)%' then impressions end) as google_other_impressions,
	-- LinkedIn
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%brand%' then impressions end) as linkedin_brand_impressions,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%thought leader%' then impressions end) as linkedin_thought_leader_impressions,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%competitor%' then impressions end) as linkedin_competitor_impressions,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%soqr%' then impressions end) as linkedin_soqr_impressions,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%solution%' then impressions end) as linkedin_solution_impressions,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%retargeting%' then impressions end) as linkedin_retargeting_impressions,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%abm%' then impressions end) as linkedin_abm_impressions,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%ppc%' then impressions end) as linkedin_ppc_impressions,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) NOT SIMILAR TO '%(ppc|thought leader|abm|retargeting|solution|soqr|competitor|brand)%' then impressions end) as linkedin_other_impressions,
	-- Meta
	SUM(CASE WHEN ads_provider = 'Meta' then impressions end) as meta_impressions,

	-- ── CLICKS (capture / intent channels) ────────────────────────────────
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%brand%' then clicks end) as google_brand_clicks,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%competitor%' then clicks end) as google_competitor_clicks,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%solution%' then clicks end) as google_solution_clicks,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%ppc%' then clicks end) as google_ppc_clicks,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%retargeting%' then clicks end) as google_retargeting_clicks,
	SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%abm%' then clicks end) as google_abm_clicks,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%competitor%' then clicks end) as linkedin_competitor_clicks,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%solution%' then clicks end) as linkedin_solution_clicks,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%ppc%' then clicks end) as linkedin_ppc_clicks,
	SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%abm%' then clicks end) as linkedin_abm_clicks,
	SUM(CASE WHEN ads_provider = 'Meta' then clicks end) as meta_clicks,

	-- ── CPM — cost per 1,000 impressions (market inflation signal) ─────────
	-- Rising CPM over time = same dollar buys fewer eyeballs = inflation covariate
	NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%brand%' then cost end), 0)
		* 1000.0 / NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%brand%' then impressions end), 0) as google_brand_cpm,
	NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%thought leader%' then cost end), 0)
		* 1000.0 / NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%thought leader%' then impressions end), 0) as google_thought_leader_cpm,
	NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%brand%' then cost end), 0)
		* 1000.0 / NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%brand%' then impressions end), 0) as linkedin_brand_cpm,
	NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%thought leader%' then cost end), 0)
		* 1000.0 / NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%thought leader%' then impressions end), 0) as linkedin_thought_leader_cpm,
	NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%abm%' then cost end), 0)
		* 1000.0 / NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%abm%' then impressions end), 0) as linkedin_abm_cpm,
	NULLIF(SUM(CASE WHEN ads_provider = 'Meta' then cost end), 0)
		* 1000.0 / NULLIF(SUM(CASE WHEN ads_provider = 'Meta' then impressions end), 0) as meta_cpm,

	-- ── CPC — cost per click (capture channel efficiency signal) ──────────
	NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%ppc%' then cost end), 0)
		/ NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%ppc%' then clicks end), 0) as google_ppc_cpc,
	NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%competitor%' then cost end), 0)
		/ NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%competitor%' then clicks end), 0) as google_competitor_cpc,
	NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%solution%' then cost end), 0)
		/ NULLIF(SUM(CASE WHEN ads_provider = 'Google' and LOWER(campaign_group) LIKE '%solution%' then clicks end), 0) as google_solution_cpc,
	NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%ppc%' then cost end), 0)
		/ NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%ppc%' then clicks end), 0) as linkedin_ppc_cpc,
	NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%solution%' then cost end), 0)
		/ NULLIF(SUM(CASE WHEN ads_provider = 'Linkedin' and LOWER(campaign_group) LIKE '%solution%' then clicks end), 0) as linkedin_solution_cpc

FROM dm_marketing.fact_ads_performance a
LEFT JOIN dw_dim.date_time d on a.date = d.date_dt
WHERE date >='2022-01-01'
GROUP BY 1
),

mqls as (
SELECT 
	month_id as month, 
	SUM(is_mql_business_new_logo) as MQLs,
    count(distinct case when is_mql_business_new_logo = 1 and lead_source_type = 'high intent' then a.id end)::float/SUM(is_mql_business_new_logo) as MQL_high_intent_rate,
    count(distinct case when is_mql_business_new_logo = 1 and segment_1b <> 'SMB' then a.id end)::float/SUM(is_mql_business_new_logo) as MQL_non_SMB_rate,
    count(distinct case when is_sql =1 and opp_type = 'New Business' and rn = 1 then opp_id end)::float/NULLIF(count(distinct case when is_mql_business_new_logo = 1  then a.id end),0) as CR_MQL_to_SQL
FROM dm_marketing.sfdc_leads a
LEFT JOIN dw_dim.date_time d on a.mql_date = d.date_dt 
WHERE mql_date >='2022-01-01'
GROUP BY 1 
),

pipeline as (
SELECT 
	month_id as month, 
	SUM(sql_new_business_pipeline) as inbound_pipeline
FROM dm_marketing.sfdc_opportunity a
LEFT JOIN dw_dim.date_time d on a.sql_transfered_date = d.date_dt 
WHERE sql_transfered_date >='2022-01-01'
AND opp_source = 'Inbound'
and opp_type ='New Business'
and is_annual = 1
GROUP BY 1 
),

arr as (
SELECT 
	month_id as month, 
	SUM(won_arr) as WON_ARR
FROM dm_marketing.sfdc_opportunity a
LEFT JOIN dw_dim.date_time d on a.opp_closed_date = d.date_dt 
WHERE opp_closed_date >='2022-01-01'
AND opp_source = 'Inbound'
and opp_type ='New Business'
and is_annual = 1
GROUP BY 1 
)

SELECT 
	*,
	case when month <= 202410 then 0 else 1 end as strategy_v2
FROM spend
FULL JOIN mqls USING (month)
FULL JOIN pipeline USING (month)
FULL JOIN arr USING (month)
where mqls > 0
ORDER BY 1 ASC 