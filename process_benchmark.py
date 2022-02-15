#!/usr/bin/env python3
import os
import json
import csv
import logging
import glob

logging.basicConfig(level=logging.INFO)

HEADER = ['input', 'throughput', 'duration']

def avg(l):
    return float(sum(l)) / float(len(l))

# def produce_csv(base_dir, output_file):
#     logging.info(f"Processing benchmark in {base_dir} and writing results to {output_file}...")
#     with open(output_file, 'w', encoding='UTF8') as f:
#         writer = csv.writer(f)
#         writer.writerow(HEADER)
#         for dir in os.listdir(base_dir):
#             if dir != "report":
#                 input_var = int(dir)
#                 sample_file = os.path.join(base_dir, dir, "new/sample.json")
#                 logging.info(f"Processing sample file {sample_file}...")
#                 with open(sample_file, 'r') as samples:
#                     data = json.load(samples)
#                     times = [float(iters) * float(t) for (iters, t) in zip(data["iters"], data["times"])]
#                     duration = avg(times)
#                     writer.writerow([ input_var, duration ])
#     logging.info(f"Finished processing benchmark in {base_dir}.")

FOUND_DATA = {}

SCALES = { 'ns': 1000000000 }

logging.info("Reading input data...")
for path in glob.glob("target/criterion/*/*/new/raw.csv"):
    logging.info(f"Processing file {path}...")
    with open(path, 'r') as raw:
        csvreader = csv.DictReader(raw)
        for row in csvreader:
            logging.debug(row)
            g = FOUND_DATA.get(row["group"], {})
            val = g.get(int(row["value"]), { "value": int(row["value"]), "throughput": int(row["throughput_num"]), "values": [] })

            v = (float(row["sample_measured_value"]) / float(row["iteration_count"])) / float(SCALES[row["unit"]])

            val["values"].append(v)

            g[int(row["value"])] = val
            FOUND_DATA[row["group"]] = g
logging.info("Done!")

logging.info("Writing results...")
for (group, values) in FOUND_DATA.items():
    file_name = f"benchmark_{group}.csv"
    logging.info(f"Writing group {group} to file {file_name}...")
    with open(file_name, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for (value, obj) in sorted(values.items()):
            samples = obj["values"]
            throughput = obj["throughput"]
            average = avg(samples)
            logging.debug(f"{value} - {throughput} - {average} - {samples}")
            writer.writerow([ value, throughput, average ])
    logging.info(f"Finished writing group {group}!")
logging.info("Done writing results!")