from pathlib import Path
import pypdf

class TextExtractor:
    """文本提取器，支持 TXT、MD、PDF"""

    def extract(self, file_path: str) -> str:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix in [".txt", ".md"]:
            return self._extract_text(path)
        elif suffix == ".pdf":
            return self._extract_pdf(path)
        else:
            raise ValueError(f"不支持的文件类型: {suffix}")

    def _extract_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _extract_pdf(self, path: Path) -> str:
        try:
            
            reader = pypdf.PdfReader(str(path))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            raise ImportError("需要安装 pypdf: pip install pypdf")