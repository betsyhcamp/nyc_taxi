print("stub commented out everything")
# from kfp import dsl
#
#
# @dsl.component(base_image="python:3.11",
#               packages_to_install=[
#                    "google-cloud-bigquery",
#                    "google-cloud-storage",
#                    "pandas",
#                    "pyarrow",
#                    ]
#               )
# def extract_db_to_bucket_component(
#    project_id: str,
#    db_location: str,
#    bucket_name: str,
#    prefix: str,
#    sql_query_str: str,
# ) -> str:
#
#    """Thin KFP wrapper that just calls the corresponding func in core/ ."""
#
#    from fcstnyctaxi.core.feature import extract_db_to_bucket
#
#
#    return extract_db_to_bucket(
#        project_id=project_id,
#        db_location=db_location,
#        bucket_name = bucket_name,
#        prefix=prefix,
#        sql_query_str=sql_query_str,
#        )
