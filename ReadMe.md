gcloud builds submit --tag gcr.io/manifest-design-452713-d9/fake-news-predictor  --project=manifest-design-452713-d9


gcloud run deploy --image gcr.io/manifest-design-452713-d9/fake-news-predictor --platform managed  --project=manifest-design-452713-d9 --allow-unauthenticated