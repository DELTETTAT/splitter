from extractor import PDFExtractor
from extractor import NameSeedExtractor
from assigner import NamePageAssigner
from splitter import PDFSplitter


def run(path: str, output_dir: str = "output"):
    extractor = PDFExtractor()
    docs = extractor.extract_pages(path)

    seeder = NameSeedExtractor()
    seed_map = seeder.extract(docs)

    assigner = NamePageAssigner(seed_map)
    assignments = assigner.assign(docs, {i:next(iter(names)) for i, names in seed_map.items()})

    splitter = PDFSplitter()
    splitter.split(docs, assignments, output_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("pdf_path")
    p.add_argument("--out", default="output")
    args = p.parse_args()
    run(args.pdf_path, args.out)