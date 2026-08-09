#!/bin/bash

input_file="uniprot_ids_unique.txt"
output_dir="uniprotkb_json_unique_outputs"
failed_log="failed_uniprot_ids.txt"

mkdir -p "$output_dir"
> "$failed_log"  # empty or create the file at the beginning

while IFS= read -r uniprot_id; do
    echo "Processing $uniprot_id..."

    # Step 1: Submit job
    response=$(curl -s --request POST 'https://rest.uniprot.org/idmapping/run' \
        --form "ids=$uniprot_id" \
        --form 'from=UniProtKB_AC-ID' \
        --form 'to=UniProtKB')

    job_id=$(echo "$response" | jq -r '.jobId')

    if [[ "$job_id" == "null" || -z "$job_id" ]]; then
        echo "  ✗ Failed to get jobId for $uniprot_id"
        echo "$uniprot_id" >> "$failed_log"
        continue
    fi

    echo "  ➤ Job ID: $job_id"

    # Step 2: Poll for status
    for i in {1..20}; do
        sleep 2
        status_response=$(curl -s "https://rest.uniprot.org/idmapping/status/$job_id")
        if echo "$status_response" | grep -q '"jobStatus":"FINISHED"'; then
            break
        elif echo "$status_response" | grep -q '"jobStatus":"FAILED"'; then
            echo "  ✗ Job failed for $uniprot_id"
            echo "$uniprot_id" >> "$failed_log"
            continue 2
        fi
    done

    # Step 3: Get result (pretty-print)
    result_url="https://rest.uniprot.org/idmapping/uniprotkb/results/$job_id"
    if ! curl -s "$result_url" | jq '.' > "$output_dir/${uniprot_id}.json"; then
        echo "  ✗ Failed to save result for $uniprot_id"
        echo "$uniprot_id" >> "$failed_log"
    else
        echo "  ✓ Saved to $output_dir/${uniprot_id}.json"
    fi

done < "$input_file"

