#!/usr/bin/env python3
"""
CS2/CS:GO Trade-Up Cache Pre-Compute Script
Efficiently pre-computes and caches trade-up analysis for all rarities.

This script removes old cache files and generates fresh cache data for all rarities
with optimized performance and detailed progress reporting.

Usage:
    python pre_compute.py
    python pre_compute.py --max_combinations 5000
    python pre_compute.py --parallel --workers 4
"""

import os
import sys
import glob
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box
import psutil

# Import from the main analyzer
from analyze_tradeups import TradeUpAnalyzer


class PreComputeManager:
    """Manages the pre-computation of trade-up cache with performance optimizations."""

    def __init__(self, skins_cache_path: str = "skins_cache", tradeup_cache_path: str = "tradeup_cache",
                 max_combinations: Optional[int] = None, parallel: bool = False, workers: int = 2):
        self.skins_cache_dir = skins_cache_path
        self.tradeup_cache_dir = tradeup_cache_path
        # TradeUpAnalyzer expects a file path for skins database
        self.skins_cache_file_path = os.path.join(
            skins_cache_path, "skins_database.json")
        self.max_combinations = max_combinations  # None means calculate dynamically
        self.parallel = parallel
        self.workers = workers
        self.console = Console()

        # All supported rarities
        self.rarities = ['Industrial', 'Mil-Spec', 'Restricted', 'Classified']
        self.stattrak_options = [False, True]

        # Performance tracking
        self.start_time = None
        self.stats = {
            'total_combinations': 0,
            'profitable_found': 0,
            'cache_files_created': 0,
            'memory_usage_mb': 0
        }

        # Initialize analyzer
        self.analyzer = None  # type: Optional[TradeUpAnalyzer]

    def clear_old_cache(self) -> None:
        """Remove all existing trade-up cache files."""
        self.console.print(
            "\n[bold red]🗑️  Clearing Old Trade-Up Cache Files[/bold red]")

        if not os.path.exists(self.tradeup_cache_dir):
            self.console.print(
                "[yellow]Trade-up cache directory doesn't exist, creating...[/yellow]")
            os.makedirs(self.tradeup_cache_dir, exist_ok=True)
            return

        # Find all cache files
        cache_pattern = os.path.join(self.tradeup_cache_dir, "tradeups_*.json")
        cache_files = glob.glob(cache_pattern)

        if not cache_files:
            self.console.print(
                "[green]✅ No old trade-up cache files found[/green]")
            return

        removed_count = 0
        failed_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Removing cache files...", total=len(cache_files))

            for cache_file in cache_files:
                try:
                    # Make file writable before removal (Windows fix)
                    if os.path.exists(cache_file):
                        os.chmod(cache_file, 0o777)
                        os.remove(cache_file)
                        removed_count += 1

                    file_name = os.path.basename(cache_file)
                    progress.update(task, advance=1,
                                    description=f"Removed {file_name}")
                    time.sleep(0.05)  # Small delay for visual feedback

                except PermissionError:
                    self.console.print(
                        f"[yellow]⚠️  Permission denied for {cache_file}, skipping...[/yellow]")
                    failed_count += 1
                except Exception as e:
                    self.console.print(
                        f"[red]❌ Failed to remove {cache_file}: {e}[/red]")
                    failed_count += 1

        if removed_count > 0:
            self.console.print(
                f"[green]✅ Removed {removed_count} old trade-up cache files[/green]")
        if failed_count > 0:
            self.console.print(
                f"[yellow]⚠️  Could not remove {failed_count} files[/yellow]")

    def calculate_dynamic_combinations(self, rarity: str, stattrak: bool) -> int:
        """Calculate total possible combinations for a given rarity/stattrak config."""
        if not self.analyzer or not self.analyzer.collection_index:
            return 999999999  # Fallback if analyzer not ready

        try:
            # Create a calculator to access the cheapest items method
            from analyze_tradeups import TradeUpCalculator
            calculator = TradeUpCalculator(self.analyzer.collection_index)

            # Get cheapest items for this rarity/stattrak combination
            cheapest_items = calculator.get_cheapest_items_per_collection_per_wear(
                rarity, stattrak)

            if not cheapest_items:
                return 0

            collections = list(cheapest_items.keys())

            # Calculate single collection combinations (10:0 ratio)
            total_single = len([collection for collection in collections
                               if any(item is not None for item in cheapest_items[collection])])

            # Calculate dual collection combinations
            total_dual = 0
            for i, collection1 in enumerate(collections):
                for j, collection2 in enumerate(collections):
                    if i >= j:  # Avoid duplicates
                        continue
                    items1_count = sum(
                        1 for item in cheapest_items[collection1] if item is not None)
                    items2_count = sum(
                        1 for item in cheapest_items[collection2] if item is not None)
                    # 9 ratio combinations per item pair: (9:1)*2 + (8:2)*2 + (7:3)*2 + (6:4)*2 + (5:5)*1 = 9
                    total_dual += items1_count * items2_count * 9

            total_combinations = total_single + total_dual
            return total_combinations

        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Could not calculate combinations for {rarity} {stattrak}: {e}[/yellow]")
            return 999999999  # Fallback to unlimited

    def initialize_analyzer(self) -> None:
        """Initialize the TradeUpAnalyzer with data loading."""
        self.console.print(
            "\n[bold blue]🔧 Initializing Trade-Up Analyzer[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            # Ensure cache directories exist with proper permissions
            init_task = progress.add_task(
                "Setting up cache directories...", total=None)
            try:
                # Create skins cache directory
                if not os.path.exists(self.skins_cache_dir):
                    os.makedirs(self.skins_cache_dir,
                                mode=0o777, exist_ok=True)

                # Create tradeup cache directory
                if not os.path.exists(self.tradeup_cache_dir):
                    os.makedirs(self.tradeup_cache_dir,
                                mode=0o777, exist_ok=True)

                # Test write access to tradeup cache directory
                test_file = os.path.join(
                    self.tradeup_cache_dir, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                progress.update(
                    init_task, description="✅ Cache directories ready")
            except Exception as e:
                self.console.print(
                    f"[red]❌ Cache directory setup failed: {e}[/red]")
                raise

            # Initialize analyzer
            analyzer_task = progress.add_task(
                "Creating analyzer instance...", total=None)
            self.analyzer = TradeUpAnalyzer(self.skins_cache_file_path)
            progress.update(analyzer_task, description="✅ Analyzer created")

            # Load data
            load_task = progress.add_task(
                "Loading CS2 skin database...", total=None)
            try:
                self.analyzer.load_data(
                    force_refresh=False, allow_consumer_inputs=True)
                progress.update(load_task, description="✅ Database loaded")
            except Exception as e:
                progress.update(
                    load_task, description=f"❌ Database load failed: {str(e)}")
                raise

        self.console.print(
            "[green]✅ Analyzer initialized successfully[/green]")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def analyze_single_configuration(self, rarity: str, stattrak: bool,
                                     progress_callback: Optional[Callable] = None) -> Dict:
        """Analyze a single rarity/stattrak configuration."""
        config_name = f"{rarity} {'StatTrak' if stattrak else 'Normal'}"

        try:
            # Track memory before analysis
            memory_before = self.get_memory_usage()

            if progress_callback:
                progress_callback(f"Starting {config_name}...")

            # Perform analysis with optimized settings
            if self.analyzer is None:
                raise ValueError("Analyzer not initialized")

            # Calculate dynamic combination limit if not specified
            if self.max_combinations is None:
                dynamic_limit = self.calculate_dynamic_combinations(
                    rarity, stattrak)
                if progress_callback:
                    progress_callback(
                        f"📊 {config_name}: {dynamic_limit:,} combinations")
            else:
                dynamic_limit = self.max_combinations

            results = self.analyzer.analyze(
                rarity=rarity,
                stattrak=stattrak,
                min_roi=0.0,  # No ROI filter for caching
                max_cost=0.0,  # No cost filter for caching
                top=1000,  # Cache more results
                use_cache=True,  # Enable cache saving for pre-compute
                max_combinations=dynamic_limit,  # Use calculated or specified limit
                debug=False  # Disable debug for performance
            )

            # Track memory after analysis
            memory_after = self.get_memory_usage()
            memory_used = memory_after - memory_before

            if progress_callback:
                progress_callback(f"✅ {config_name} complete")

            return {
                'config': config_name,
                'rarity': rarity,
                'stattrak': stattrak,
                'results_count': len(results),
                'profitable_count': len([r for r in results if r.roi > 0]),
                'memory_used_mb': memory_used,
                'success': True
            }

        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ {config_name} failed: {str(e)}")

            return {
                'config': config_name,
                'rarity': rarity,
                'stattrak': stattrak,
                'results_count': 0,
                'profitable_count': 0,
                'memory_used_mb': 0,
                'success': False,
                'error': str(e)
            }

    def run_sequential(self) -> List[Dict]:
        """Run pre-computation sequentially with rich progress display."""
        self.console.print(
            "\n[bold cyan]🔄 Sequential Pre-Computation Started[/bold cyan]")

        if self.max_combinations is None:
            self.console.print(
                "[bold green]🧮 DYNAMIC MODE: Calculating exact combinations per rarity based on available skins/variants[/bold green]")
        else:
            self.console.print(
                f"[yellow]📊 Fixed limit: {self.max_combinations:,} combinations per config[/yellow]")

        total_configs = len(self.rarities) * len(self.stattrak_options)
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[config]}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        ) as progress:

            # Create main progress task
            main_task = progress.add_task(
                "Overall Progress",
                total=total_configs,
                config="Initializing..."
            )

            for i, rarity in enumerate(self.rarities):
                for j, stattrak in enumerate(self.stattrak_options):
                    config_name = f"{rarity} {'StatTrak' if stattrak else 'Normal'}"

                    # Update progress
                    progress.update(
                        main_task,
                        advance=0,
                        config=f"Processing {config_name}"
                    )

                    # Analyze configuration
                    def update_progress(msg):
                        progress.update(main_task, config=msg)

                    result = self.analyze_single_configuration(
                        rarity, stattrak, update_progress
                    )
                    results.append(result)

                    # Update stats
                    self.stats['profitable_found'] += result['profitable_count']
                    self.stats['memory_usage_mb'] = max(
                        self.stats['memory_usage_mb'],
                        result['memory_used_mb']
                    )

                    # Advance progress
                    progress.update(main_task, advance=1)

                    # Brief pause for visual feedback
                    time.sleep(0.1)

        return results

    def run_parallel(self) -> List[Dict]:
        """Run pre-computation in parallel with thread pool."""
        self.console.print(
            f"\n[bold cyan]⚡ Parallel Pre-Computation Started ({self.workers} workers)[/bold cyan]")

        if self.max_combinations is None:
            self.console.print(
                "[bold green]🧮 DYNAMIC MODE: Calculating exact combinations per rarity based on available skins/variants[/bold green]")
        else:
            self.console.print(
                f"[yellow]📊 Fixed limit: {self.max_combinations:,} combinations per config[/yellow]")

        total_configs = len(self.rarities) * len(self.stattrak_options)
        results = []
        completed_count = 0

        # Create all configuration tasks
        tasks = []
        for rarity in self.rarities:
            for stattrak in self.stattrak_options:
                tasks.append((rarity, stattrak))

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[status]}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        ) as progress:

            main_task = progress.add_task(
                "Parallel Processing",
                total=total_configs,
                status="Starting workers..."
            )

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                future_to_config = {}
                for rarity, stattrak in tasks:
                    future = executor.submit(
                        self.analyze_single_configuration,
                        rarity, stattrak, None
                    )
                    future_to_config[future] = (rarity, stattrak)

                # Process completed tasks
                for future in as_completed(future_to_config):
                    rarity, stattrak = future_to_config[future]
                    config_name = f"{rarity} {'StatTrak' if stattrak else 'Normal'}"

                    try:
                        result = future.result()
                        results.append(result)

                        # Update stats
                        self.stats['profitable_found'] += result['profitable_count']
                        self.stats['memory_usage_mb'] = max(
                            self.stats['memory_usage_mb'],
                            result['memory_used_mb']
                        )

                        completed_count += 1

                        # Update progress
                        progress.update(
                            main_task,
                            advance=1,
                            status=f"Completed {config_name} ({completed_count}/{total_configs})"
                        )

                    except Exception as e:
                        self.console.print(
                            f"[red]❌ {config_name} failed: {e}[/red]")
                        completed_count += 1
                        progress.update(main_task, advance=1)

        return results

    def display_results_summary(self, results: List[Dict]) -> None:
        """Display a comprehensive summary of pre-computation results."""

        # Calculate summary statistics
        if self.start_time is None:
            self.start_time = time.time()
        total_time = time.time() - self.start_time
        successful_configs = [r for r in results if r['success']]
        failed_configs = [r for r in results if not r['success']]
        total_results = sum(r['results_count'] for r in successful_configs)
        total_profitable = sum(r['profitable_count']
                               for r in successful_configs)

        # Calculate average success rate and ROI across all successful configs
        avg_success_rate = 0.0
        avg_roi = 0.0
        total_candidates = 0
        for r in successful_configs:
            # Each result is for a config, and has a list of candidates
            # We need to get the average success rate and ROI for all candidates in all configs
            if r['results_count'] > 0:
                # If available, get the average from the candidates
                if 'candidates' in r:
                    avg_success_rate += sum(
                        c.success_rate for c in r['candidates'] if c.success_rate is not None)
                    avg_roi += sum(c.roi for c in r['candidates'])
                    total_candidates += len(r['candidates'])
                else:
                    # Fallback: use config-level success_rate and roi if present
                    avg_success_rate += r.get('success_rate',
                                              0) * r['results_count']
                    avg_roi += r.get('roi', 0) * r['results_count']
                    total_candidates += r['results_count']

        avg_success_rate = (avg_success_rate / max(total_candidates, 1)) * 100
        avg_roi = (avg_roi / max(total_candidates, 1)) * 100

        # Create summary panel
        summary_table = Table(title="Pre-Computation Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bold green")

        summary_table.add_row("📊 Total Time", f"{total_time:.1f} seconds")
        summary_table.add_row("✅ Successful Configs",
                              f"{len(successful_configs)}/8")
        summary_table.add_row("❌ Failed Configs", str(len(failed_configs)))
        summary_table.add_row("🔍 Total Results", f"{total_results:,}")
        summary_table.add_row("💰 Profitable Results", f"{total_profitable:,}")
        summary_table.add_row("🎲 Avg Success Rate", f"{avg_success_rate:.2f}%")
        summary_table.add_row("📈 Avg Profitability (ROI)", f"{avg_roi:.2f}%")
        summary_table.add_row("💾 Peak Memory Usage",
                              f"{self.stats['memory_usage_mb']:.1f} MB")
        summary_table.add_row("⚡ Avg Time per Config",
                              f"{total_time/len(results):.1f} sec")

        self.console.print("\n")
        self.console.print(
            Panel(summary_table, title="🎯 Final Results", border_style="green"))

        # Detailed results table
        if successful_configs:
            details_table = Table(
                title="Configuration Details", box=box.SIMPLE)
            details_table.add_column("Configuration", style="bold yellow")
            details_table.add_column("Results", justify="right")
            details_table.add_column(
                "Profitable", justify="right", style="green")
            details_table.add_column(
                "Avg Success Rate", justify="right", style="blue")
            details_table.add_column("Avg ROI", justify="right", style="cyan")
            details_table.add_column(
                "Memory", justify="right", style="magenta")

            for result in successful_configs:
                # Calculate per-config averages
                if result['results_count'] > 0 and 'candidates' in result:
                    config_success_rate = sum(
                        c.success_rate for c in result['candidates'] if c.success_rate is not None) / max(len(result['candidates']), 1) * 100
                    config_roi = sum(
                        c.roi for c in result['candidates']) / max(len(result['candidates']), 1) * 100
                else:
                    config_success_rate = (result.get('success_rate', 0)) * 100
                    config_roi = (result.get('roi', 0)) * 100
                details_table.add_row(
                    result['config'],
                    f"{result['results_count']:,}",
                    f"{result['profitable_count']:,}",
                    f"{config_success_rate:.2f}%",
                    f"{config_roi:.2f}%",
                    f"{result['memory_used_mb']:.1f} MB"
                )

            self.console.print("\n")
            self.console.print(details_table)

        # Show any failures
        if failed_configs:
            self.console.print(
                "\n[bold red]❌ Failed Configurations:[/bold red]")
            for result in failed_configs:
                self.console.print(
                    f"  • {result['config']}: {result.get('error', 'Unknown error')}")

    def run(self) -> None:
        """Execute the complete pre-computation process."""
        self.start_time = time.time()

        # Display header
        max_combinations_display = "Dynamic (calculated per rarity)" if self.max_combinations is None else f"{self.max_combinations:,}"

        header = Panel(
            "[bold cyan]CS2/CS:GO Trade-Up Cache Pre-Compute[/bold cyan]\n"
            f"🔧 Max Combinations: [yellow]{max_combinations_display}[/yellow]\n"
            f"⚡ Parallel Mode: [yellow]{'Enabled' if self.parallel else 'Disabled'}[/yellow]\n"
            f"👥 Workers: [yellow]{self.workers if self.parallel else 1}[/yellow]\n"
            f"📁 Skins Cache: [yellow]{self.skins_cache_dir}[/yellow]\n"
            f"📁 Trade-up Cache: [yellow]{self.tradeup_cache_dir}[/yellow]",
            title="🚀 Configuration",
            border_style="blue"
        )
        self.console.print(header)

        try:
            # Step 1: Clear old cache
            self.clear_old_cache()

            # Step 2: Initialize analyzer
            self.initialize_analyzer()

            # Step 3: Run pre-computation
            if self.parallel:
                results = self.run_parallel()
            else:
                results = self.run_sequential()

            # Step 4: Display results
            self.display_results_summary(results)

            # Final success message
            self.console.print(
                "\n[bold green]🎉 Pre-computation completed successfully![/bold green]")

        except KeyboardInterrupt:
            self.console.print(
                "\n[yellow]⚠️  Pre-computation interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(
                f"\n[bold red]💥 Pre-computation failed: {e}[/bold red]")
            sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CS2/CS:GO Trade-Up Cache Pre-Compute Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pre_compute.py                                    # Basic pre-compute
  python pre_compute.py --max_combinations 5000           # Limit combinations
  python pre_compute.py --parallel --workers 4            # Parallel processing
  python pre_compute.py --skins_cache_path custom_skins --tradeup_cache_path custom_tradeups  # Custom paths
        """
    )

    parser.add_argument(
        '--max_combinations',
        type=int,
        default=None,
        help='Maximum combinations per rarity/stattrak config (default: unlimited - analyzes ALL combinations)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing for faster computation'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of worker threads for parallel processing (default: 2)'
    )

    parser.add_argument(
        '--skins_cache_path',
        default='skins_cache',
        help='Path to skins cache directory (default: skins_cache)'
    )

    parser.add_argument(
        '--tradeup_cache_path',
        default='tradeup_cache',
        help='Path to trade-up cache directory (default: tradeup_cache)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.workers < 1:
        print("Error: Number of workers must be at least 1")
        sys.exit(1)

    if args.max_combinations is not None and args.max_combinations < 10:
        print("Error: Max combinations must be at least 10 (or None for unlimited)")
        sys.exit(1)

    # Create and run pre-compute manager
    manager = PreComputeManager(
        skins_cache_path=args.skins_cache_path,
        tradeup_cache_path=args.tradeup_cache_path,
        max_combinations=args.max_combinations,
        parallel=args.parallel,
        workers=args.workers
    )

    manager.run()


if __name__ == "__main__":
    main()
