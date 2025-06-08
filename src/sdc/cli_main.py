from .sdc_main import world, system

SIM_TIME_STEP = 1.0 / 120.0
MAX_TICKS = 2400


def run_cli():
    world().run(system(), sim_time_step=SIM_TIME_STEP, max_ticks=MAX_TICKS)
