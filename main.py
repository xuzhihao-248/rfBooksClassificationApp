import argparse

from src.classifier import BookClassifier


def main():
    parser = argparse.ArgumentParser(description="Book classification CLI")
    parser.add_argument("--model", choices=["rf", "transformer"], default="rf",
                        help="Model to use: rf (Word2Vec+RandomForest) or transformer")
    args = parser.parse_args()

    print(f"Loading {args.model} model...")
    try:
        classifier = BookClassifier(model_type=args.model)
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run 'python train.py --model {args.model}' first.")
        return

    print("Book classification ready. Type 'quit' to exit.\n")

    while True:
        try:
            title = input("请输入书名: ").strip()
            if title.lower() == "quit":
                print("Goodbye!")
                break
            if not title:
                print("书名不能为空，请重新输入。\n")
                continue

            description = input("请输入内容简介（可选，按回车跳过）: ").strip()
            if description.lower() == "quit":
                print("Goodbye!")
                break

            result = classifier.predict(title, description)

            print()
            print("=" * 50)
            print(f"预测结果: {result['class_code']} ({result['class_name']})")
            print(f"置信度: {result['confidence']}")
            print("=" * 50)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
