import asyncio
from pathlib import Path
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def main():

    from src.preprocessing.preprocessor_factory import PreprocessorFactory
    from config.logging_config import configure_logging

    configure_logging("WARNING")

    preprocessor = PreprocessorFactory.create(worker="minimal")
    file_name = "zbirnik_final_2022_v6-7ok_1"

    folder = "python-3.13.8-docs-text"

    root_path = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + f"\\data\\raw\\"

    print("Test: processing document")

    sum_chars = 0

    try:
        start_time = time.time()
        for file in get_files(Path(root_path + folder)):
            result = preprocessor.process_document(file, )
            sum_chars += sum(len(chunk.text) for chunk in result.chunks)

        end_time = time.time()
        print(
            f"Processed documents in {end_time - start_time:.2f} seconds, total characters: {sum_chars}"
        )

    except Exception as e:
        print(f"Error during async processing: {e}")
        raise


if __name__ == '__main__':
    main()
    # asyncio.run(main())
