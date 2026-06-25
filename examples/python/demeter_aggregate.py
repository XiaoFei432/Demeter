from MARL.driver import MultiProcessFunctionDriver, cpu_cores_from_env


def aggregate_chunk(rows):
    return sum(hash(str(row)) & 0xFFFF for row in rows)


def handle(records):
    driver = MultiProcessFunctionDriver(cpu_cores_from_env())
    return sum(driver.run(aggregate_chunk, records))
