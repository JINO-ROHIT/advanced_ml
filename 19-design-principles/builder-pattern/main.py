# its like making a matpllotlib type plot or numpy chains

from htmlbuilder import HTMLBuilder


def main() -> None:
    # --- Build UI Page ---
    builder = HTMLBuilder()
    page = (
        builder.set_title("Builder Pattern UI")
        .add_header("Hello from Python!", level=1)
        .add_paragraph("This page was generated using the Builder Pattern.")
        .add_button("Visit ArjanCodes", onclick="https://www.arjancodes.com")
        .build()
    )

    # --- Write to HTML File ---
    file_path = "page.html"
    with open(file_path, "w") as f:
        f.write(page.render())

    print("HTML page written to 'page.html'")


if __name__ == "__main__":
    main()