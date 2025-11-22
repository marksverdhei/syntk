import sys
import typer

app = typer.Typer(
    help="Syntk - Toolkit for synthetic data generation and processing",
    no_args_is_help=True
)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def column(ctx: typer.Context):
    """Fill column values using LLM (equivalent to lmfill)."""
    from syntk import lmfill

    # Restore original sys.argv to pass all arguments to lmfill
    # Remove 'syntk' and 'column' from argv, keeping the rest
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + ctx.args

    try:
        lmfill.main()
    finally:
        sys.argv = original_argv


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Syntk - Toolkit for synthetic data generation and processing"""
    if ctx.invoked_subcommand is None:
        typer.echo("Use 'syntk column' or 'syntk --help' for more information")


def main():
    app()


if __name__ == "__main__":
    main()
