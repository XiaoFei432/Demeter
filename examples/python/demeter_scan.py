from MARL.driver import MultiProcessFunctionDriver, cpu_cores_from_env


def scan_chunk(rows):
    total = 0
    for row in rows:
        total += len(str(row))
    return total


def handle(records):
    driver = MultiProcessFunctionDriver(cpu_cores_from_env())
    return sum(driver.run(scan_chunk, records))
