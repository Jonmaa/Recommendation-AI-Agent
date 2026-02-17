# ============================================================
# main.py - Interactive CLI for the Product Recommendation Agent
# ============================================================
# Run this file to start the recommendation system.
# Usage:
#   python main.py
# ============================================================

import sys
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

from database import get_all_products, get_product, get_user_by_name, PRODUCTS
from vector_store import ProductVectorStore
from recommendation_agent import RecommendationAgent


console = Console()


def display_product_catalog() -> None:
    """Print the full product catalog as a formatted table."""
    table = Table(title="Product Catalog", show_lines=True)
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Name", style="bold white", width=35)
    table.add_column("Category", style="green", width=12)
    table.add_column("Price", style="yellow", justify="right", width=10)
    table.add_column("Description", style="dim", width=55)

    for product in get_all_products():
        table.add_row(
            product.id,
            product.name,
            product.category,
            f"${product.price:.2f}",
            product.description[:55] + "..." if len(product.description) > 55 else product.description,
        )

    console.print(table)


def display_help() -> None:
    """Show available commands."""
    help_text = """
**Available Commands:**

| Command | Description |
|---------|-------------|
| `catalog` | Show all products |
| `buy <product_id>` | Buy a product and get recommendations |
| `purchases` | Show your purchase history |
| `search <query>` | Search for products by keyword |
| `chat <message>` | Ask the agent a question |
| `reset` | Reset conversation history |
| `help` | Show this help |
| `quit` / `exit` | Exit the program |

**Example:**
```
buy P001      → Buys Nike Air Max 90 and gets recommendations
search yoga   → Searches for yoga-related products
```
"""
    console.print(Markdown(help_text))


def search_products(query: str, vector_store: ProductVectorStore) -> None:
    """Search products using FAISS semantic similarity."""
    results = vector_store.search_similar_products(query, top_k=5)

    if not results:
        console.print("[yellow]No matching products found.[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}'", show_lines=True)
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Name", style="bold white", width=35)
    table.add_column("Category", style="green", width=12)
    table.add_column("Price", style="yellow", justify="right", width=10)
    table.add_column("Similarity", style="magenta", justify="right", width=10)

    for pid, score, _ in results:
        p = PRODUCTS[pid]
        table.add_row(p.id, p.name, p.category, f"${p.price:.2f}", f"{score:.3f}")

    console.print(table)


def main():
    """Main entry point for the interactive recommendation agent."""
    console.print(
        Panel.fit(
            "[bold blue]AI Product Recommendation Agent[/bold blue]\n"
            "[dim]Powered by RAG + FAISS + Groq[/dim]",
            border_style="blue",
        )
    )

    # Check for Groq API key
    if not os.getenv("GROQ_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] GROQ_API_KEY not found.\n"
            "Set it as an environment variable or create a .env file.\n"
            "Get a free key at: [cyan]https://console.groq.com[/cyan]"
        )
        sys.exit(1)

    # Initialize FAISS vector store (builds embeddings on startup)
    console.print("[dim]Initializing FAISS vector store and generating embeddings...[/dim]")
    vector_store = ProductVectorStore()

    # Initialize the recommendation agent
    agent = RecommendationAgent(vector_store)

    # User session — look up or create user by name
    username = Prompt.ask("\n[bold]Enter your name", default="User")

    existing_user = get_user_by_name(username)
    if existing_user:
        purchased_names = [PRODUCTS[pid].name for pid in existing_user.purchased_product_ids if pid in PRODUCTS]
        console.print(
            f"\nWelcome back, [bold green]{username}[/bold green]! "
            f"(ID: {existing_user.user_id})\n"
            f"[dim]Your previous purchases: {', '.join(purchased_names)}[/dim]\n"
        )
    else:
        console.print(f"\nWelcome, [bold green]{username}[/bold green]! You're a new user.\n")

    console.print(f"Type [cyan]help[/cyan] for commands.\n")

    # --- Interactive Loop ---
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>>>[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        command = user_input.lower().split()[0]
        args = user_input[len(command):].strip()

        # ---- Command Router ----
        if command in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break

        elif command == "help":
            display_help()

        elif command == "catalog":
            display_product_catalog()

        elif command == "purchases":
            user = get_user_by_name(username)
            if not user or not user.purchased_product_ids:
                console.print("[yellow]You haven't bought anything yet.[/yellow]")
            else:
                table = Table(title=f"{username}'s Purchases", show_lines=True)
                table.add_column("ID", style="cyan", width=6)
                table.add_column("Name", style="bold white", width=35)
                table.add_column("Category", style="green", width=12)
                table.add_column("Price", style="yellow", justify="right", width=10)
                total = 0.0
                for pid in user.purchased_product_ids:
                    p = PRODUCTS.get(pid)
                    if p:
                        table.add_row(p.id, p.name, p.category, f"${p.price:.2f}")
                        total += p.price
                table.add_row("", "[bold]TOTAL[/bold]", "", f"[bold]${total:.2f}[/bold]")
                console.print(table)

        elif command == "buy":
            if not args:
                console.print("[yellow]Usage: buy <product_id>  (e.g., buy P001)[/yellow]")
                continue

            product_id = args.upper()
            product = get_product(product_id)

            if not product:
                console.print(f"[red]Product '{product_id}' not found. Use 'catalog' to see all products.[/red]")
                continue

            console.print(
                f"\n[bold green]Purchase confirmed![/bold green] "
                f"You bought: [bold]{product.name}[/bold] (${product.price:.2f})\n"
            )
            console.print("[dim]Generating recommendations...[/dim]\n")

            try:
                recommendation = agent.recommend_after_purchase(username, product_id)
                console.print(Panel(
                    Markdown(recommendation),
                    title="[bold blue]AI Recommendation[/bold blue]",
                    border_style="blue",
                ))
            except Exception as e:
                console.print(f"[red]Error generating recommendation: {e}[/red]")

        elif command == "search":
            if not args:
                console.print("[yellow]Usage: search <query>  (e.g., search running shoes)[/yellow]")
                continue
            search_products(args, vector_store)

        elif command == "chat":
            if not args:
                console.print("[yellow]Usage: chat <message>  (e.g., chat Tell me more about the Garmin watch)[/yellow]")
                continue

            try:
                reply = agent.chat(args)
                console.print(Panel(
                    Markdown(reply),
                    title="[bold blue]Agent[/bold blue]",
                    border_style="blue",
                ))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

        elif command == "reset":
            agent.reset_conversation()
            console.print("[green]Conversation history cleared.[/green]")

        else:
            # Treat unknown input as a chat message
            try:
                reply = agent.chat(user_input)
                console.print(Panel(
                    Markdown(reply),
                    title="[bold blue]Agent[/bold blue]",
                    border_style="blue",
                ))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
