docker  run \
    --name=squad_data_dl \
    -v ./data:/data:rw \
    --network container:deh_rag_api \
    -e API_ANSWER_ENDPOINT='deh_rag_api:8080' \
    -e OLLAMA_URL='http://172.17.0.1:7869' \
    --rm \
    deh_measurement \
    ragas_evaluation.py --qas_file /data/qas/squad_qas.tsv --evaluation_folder /data/evaluation --sample_size 3