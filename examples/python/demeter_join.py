from MARL.driver import MultiProcessFunctionDriver, cpu_cores_from_env


def join_chunk(rows):
    return [row for row in rows if hash(str(row)) % 3 == 0]


def handle(records):
    driver = MultiProcessFunctionDriver(cpu_cores_from_env())
    chunks = driver.run(join_chunk, records)
    out = []
    for chunk in chunks:
        out.extend(chunk)
    return out
