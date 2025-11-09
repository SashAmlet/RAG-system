import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():

    from src.preprocessing.preprocessor import Preprocessor
    from config.logging_config import configure_logging

    configure_logging("INFO")

    preprocessor = Preprocessor()

    file_name = "zbirnik_final_2022_v6-7ok_1"

    path = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + f"\\data\\raw\\{file_name}.pdf"

    print("Async test: processing document")
    try:
        # result = await preprocessor.process_document_async(path, use_marker=True)
        result2 = preprocessor.process_document(path, use_marker=False)
        # await preprocessor.save_document_async(path, result.processed_text)
        preprocessor.save_document(path, result2.processed_text)
    except Exception as e:
        print(f"Error during async processing: {e}")
        raise


if __name__ == '__main__':
    main()
    # asyncio.run(main())
