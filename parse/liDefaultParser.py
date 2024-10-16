import json

from llama_index.core import SimpleDirectoryReader

from parse.parser import Parser


class LiDefaultParser(Parser):
    show_progress = False

    @property
    def supported_input_formats(self):
        return ["epub", "docx", "doc", "pdf", "csv", "hwp", "ipynb", "jpeg", "jpg", "mbox", "md", "mp3", "mp4", "png", "ppt", "pptx", "pptm", "xlx", "xlsx"]
    
    def parse_file(self, input_file: str, output_file: str) -> bool:
        documents = SimpleDirectoryReader(input_files=[input_file]).load_data(show_progress=self.show_progress)
        if len(documents) == 0:
            return False
        documents_json = [doc.to_dict() for doc in documents]
        with open(output_file, 'w') as f:
            f.write(json.dumps(documents_json, ensure_ascii=False, indent=2))
        return True