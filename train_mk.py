

# Овозможува користење на нови type-hint синтакси (на пр. `str | None`) и во
# постари верзии на Python
from __future__ import annotations

# argparse -- за парсирање на аргументи од командна линија
import argparse
# glob -- за пребарување датотеки по шаблон (пр. наоѓање на checkpoint-и)
import glob
# os -- за работа со патеки и директориуми
import os

# NumPy -- за нумерички операции (пр. clip)
import numpy as np
# SAC -- алгоритмот Soft Actor-Critic од Stable-Baselines3
from stable_baselines3 import SAC
# Callback-класи: базен callback, евалуација, листа од callback-и, checkpoint зачувување
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList, CheckpointCallback
# Помошна функција за креирање на векторизирани (паралелни) опкружувања
from stable_baselines3.common.env_util import make_vec_env
# Два типа на векторизирано опкружување: паралелно (subprocess) и секвенцијално (во иста нишка)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Нашата прилагодена околина и стандардните тежини на наградата (од ant_reward_env.py)
from ant_reward_env import make_ant_env, DEFAULT_WEIGHTS

# Секој клуч (тежина на награда) има свој пар (onset_frac, ramp_frac): тежината
# е 0 пред onset_frac, линеарно расте од 0 до целосната (базна) вредност во
# текот на следните ramp_frac дел од вкупниот напредок на тренингот, а потоа
# останува на целосна вредност до крајот на тренингот.
# ---------------------------------------------------------------------------

KEY_SCHEDULE = {
    # Активни на целосна јачина од чекор 0: поттиците за брзина, И двете
    # безбедносно-критични казни (паѓање, застој) кои никогаш не смеат да
    # имаат "слободен" прозорец во кој неподвижноста/паѓањето изгледаат
    # бесплатни.
    "w_vel": (0.00, 0.00),
    "w_survival": (0.00, 0.00),
    "w_avg_vel": (0.00, 0.00),
    "w_fall": (0.00, 0.00),
    "w_idle": (0.00, 0.00),
    # Фаза 2 (20% -> 55%): термините за стабилност постепено се воведуваат.
    "w_height": (0.20, 0.35),
    "w_orientation": (0.20, 0.35),
    "w_energy": (0.20, 0.35),
    # Фаза 3 (55% -> 100%): регулаторите за прецизирање на одот се воведуваат.
    "w_joint_limit": (0.55, 0.45),
    "w_joint_vel": (0.55, 0.45),
    "w_symmetry": (0.55, 0.45),
    "w_smooth": (0.55, 0.45),
}

# едноставен hard code за поставување на тежините
def interpolate_weights(progress: float, base_weights: dict) -> dict:
    progress = float(np.clip(progress, 0.0, 1.0))
    weights = {}
    # Поминуваме низ секоја базна (целосна) тежина
    for k, full_value in base_weights.items():
        # Го земаме распоредот (onset, ramp) за овој клуч; ако нема во KEY_SCHEDULE -> веднаш целосна вредност
        onset, ramp = KEY_SCHEDULE.get(k, (0.0, 0.0))
        if progress <= onset:
            # Пред почетокот на воведувањето -- тежината е нула
            weights[k] = 0.0
        elif ramp <= 1e-8 or progress >= onset + ramp:
            # Нема период на постепено воведување, или веќе го поминале -- целосна вредност
            weights[k] = full_value
        else:
            # Во средината на постепеното воведување -- линеарна интерполација од 0 до full_value
            alpha = (progress - onset) / ramp
            weights[k] = alpha * full_value
    return weights


class CurriculumCallback(BaseCallback):
    """Периодично ги проследува ажурираните тежини на наградата до секое под-опкружување."""

    def __init__(self, total_timesteps: int, base_weights: dict, update_every: int = 2048, verbose: int = 0):
        # Иницијализација на родителската BaseCallback класа
        super().__init__(verbose)
        self.total_timesteps = total_timesteps  # вкупен број чекори планирани за тренинг (за пресметка на напредок)
        self.base_weights = base_weights        # целосните (крајни) тежини кон кои се стреми curriculum-от
        self.update_every = update_every        # колку чекори меѓу секое ажурирање на тежините
        self._last_update_step = -1             # чекор кога последен пат биле ажурирани тежините

    def _on_step(self) -> bool:
        # Проверува дали поминало доволно чекори од последното ажурирање
        if self.num_timesteps - self._last_update_step >= self.update_every:
            # Пресметка на напредокот на тренингот како дел [0,1]
            progress = min(self.num_timesteps / self.total_timesteps, 1.0)
            # Пресметка на новите тежини според curriculum распоредот
            weights = interpolate_weights(progress, self.base_weights)
            # Ги праќаме новите тежини до сите паралелни под-опкружувања (env_method повикува
            # set_reward_weights() во секое од нив)
            self.training_env.env_method("set_reward_weights", weights)
            self._last_update_step = self.num_timesteps
            if self.verbose:
                print(f"[curriculum] step={self.num_timesteps} progress={progress:.2f} weights={weights}")
        # Враќа True за да продолжи тренингот (False би го прекинало предвреме)
        return True


# ---------------------------------------------------------------------------
# Влезна точка за тренинг
# ---------------------------------------------------------------------------

def build_env_fn():
    # Враќа функција без аргументи што креира ново опкружување -- потребно
    # за make_vec_env, кој за секое паралелно опкружување повикува ваква функција
    def _init():
        return make_ant_env()
    return _init


def find_latest_checkpoint(checkpoint_dir: str, name_prefix: str = "ckpt") -> str | None:
    # Шаблон за пребарување на checkpoint-датотеки со модел (пр. ckpt_500000_steps.zip)
    pattern = os.path.join(checkpoint_dir, f"{name_prefix}_*_steps.zip")

    candidates = [
        p for p in glob.glob(pattern)
        if "_replay_buffer_" not in os.path.basename(p) and "_vecnormalize_" not in os.path.basename(p)
    ]
    if not candidates:
        # Нема пронајдени checkpoint-и
        return None
    # Ги сортираме по бројот на чекори (извлечен од името на датотеката) за да го најдеме најновиот
    candidates.sort(key=lambda p: int(os.path.basename(p).split("_")[-2]))
    return candidates[-1]


def replay_buffer_path_for(model_path: str, name_prefix: str = "ckpt") -> str:
    """Дадена патека до checkpoint на модел како '.../ckpt_2000000_steps.zip', ја враќа
    патеката што CheckpointCallback ја користел за соодветниот replay буфер:
    '.../ckpt_replay_buffer_2000000_steps.pkl'."""
    # Го извлекуваме бројот на чекори од името на датотеката на моделот
    steps = os.path.basename(model_path).split("_")[-2]
    # Ја составуваме соодветната патека за replay буферот, во истиот директориум
    return os.path.join(os.path.dirname(model_path), f"{name_prefix}_replay_buffer_{steps}_steps.pkl")


def main():
    # Дефинирање на сите аргументи што може да се проследат преку командна линија
    parser = argparse.ArgumentParser(description="Train SAC on custom-reward Ant-v5 with curriculum learning.")
    parser.add_argument("--timesteps", type=int, default=20_000_000,
                         help="Total training timesteps. Default is intentionally large -- "
                              "a stable Ant gait typically needs many millions of steps. "
                              "Interrupt and resume with --resume as needed.")
    # ^ Вкупен број чекори за тренинг. Намерно голема стандардна вредност --
    #   стабилен од на мравката обично бара милиони чекори. Прекини и
    #   продолжи со --resume по потреба.
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments.")
    # ^ Број на паралелни опкружувања.
    parser.add_argument("--eval-freq", type=int, default=20_000, help="Timesteps between evaluations (total, not per-env).")
    # ^ Чекори меѓу евалуациите (вкупно, не по опкружување).
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    # ^ Број на епизоди по евалуација.
    parser.add_argument("--checkpoint-freq", type=int, default=200_000,
                         help="Timesteps between full checkpoints (model + replay buffer).")
    # ^ Чекори меѓу целосни checkpoint-и (модел + replay буфер).
    parser.add_argument("--out-dir", type=str, default="./runs/ant_sac")
    # ^ Излезен директориум за логови, checkpoint-и и модели.
    parser.add_argument("--seed", type=int, default=0)
    # ^ Семе за случајност (за репродуцибилност).
    parser.add_argument("--curriculum-update-every", type=int, default=2048,
                         help="Timesteps between curriculum weight updates.")
    # ^ Чекори меѓу ажурирања на тежините преку curriculum.
    parser.add_argument("--use-subproc", action="store_true",
                         help="Use SubprocVecEnv instead of DummyVecEnv. Recommended whenever "
                              "n-envs > 1 -- DummyVecEnv runs everything on one core.")
    # ^ Користи SubprocVecEnv (вистински паралелни процеси) наместо DummyVecEnv
    #   (сè на едно јадро). Препорачано кога n-envs > 1.
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint .zip to resume from. If omitted, the script "
                              "will look for one automatically under <out-dir>/checkpoints/.")
    # ^ Патека до checkpoint (.zip) од каде да се продолжи. Ако не е дадена,
    #   скриптата автоматски бара најнов checkpoint во <out-dir>/checkpoints/.
    parser.add_argument("--no-sde", action="store_true",
                         help="Disable gSDE and fall back to SAC's default per-step Gaussian "
                              "exploration noise. Not recommended -- see module docstring.")
    # ^ Оневозможи gSDE и врати се на стандардниот Гаусов шум по чекор кај
    #   SAC. Не е препорачано -- види го докстрингот на модулот.
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
                         help="Replay buffer capacity (transitions). Lower this if you're "
                              "memory/disk constrained -- each transition costs roughly "
                              "(obs_dim*2 + action_dim + 2) * 4 bytes.")
    # ^ Капацитет на replay буферот (транзакции). Намали го ако немаш доволно
    #   меморија/диск -- секоја транзакција чини околу (obs_dim*2 + action_dim + 2) * 4 бајти.
    args = parser.parse_args()

    # Креирање на излезните директориуми (ако веќе постојат, не фрла грешка)
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Избор на класа за векторизирано опкружување: SubprocVecEnv (паралелни процеси)
    # само ако е побарано И има повеќе од 1 опкружување; инаку DummyVecEnv
    vec_env_cls = SubprocVecEnv if (args.use_subproc and args.n_envs > 1) else DummyVecEnv

    # Креирање на векторизирано опкружување за тренинг (n_envs паралелни копии)
    train_env = make_vec_env(
        build_env_fn(),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
        monitor_dir=log_dir,
    )

    # Одделно опкружување (само 1 копија, секогаш DummyVecEnv) за евалуација на моделот
    eval_env = make_vec_env(
        build_env_fn(),
        n_envs=1,
        seed=args.seed + 1000,
        vec_env_cls=DummyVecEnv,
        monitor_dir=os.path.join(log_dir, "eval"),
    )

    # --- Одредување на патеката за продолжување (изречно --resume, или автоматско наоѓање на најнов checkpoint) ---
    resume_path = args.resume
    if resume_path is None:
        # Ако не е дадена патека, бараме автоматски најнов checkpoint во директориумот
        auto = find_latest_checkpoint(checkpoint_dir)
        if auto is not None:
            print(f"Found existing checkpoint, resuming from: {auto}")
            resume_path = auto

    if resume_path is not None:
        # Продолжуваме тренинг: вчитуваме го зачуваниот модел, поврзан со тековното train_env
        model = SAC.load(resume_path, env=train_env)
        # Ја наоѓаме соодветната патека за replay буферот
        replay_buffer_path = replay_buffer_path_for(resume_path)
        if os.path.exists(replay_buffer_path):
            # Вчитуваме го replay буферот (историјата на искуства) ако постои
            model.load_replay_buffer(replay_buffer_path)
            print(f"Loaded replay buffer from: {replay_buffer_path}")
        else:
            # Ако нема зачуван replay буфер, продолжуваме со празен -- ќе се
            # пополни повторно преку learning_starts пред следните градиент-ажурирања
            print("No matching replay buffer found -- resuming with an empty buffer "
                  "(learning_starts will re-populate it before further gradient updates).")
        # Не го ресетираме бројачот на чекори, за да продолжи од каде застанал
        reset_num_timesteps = False
    else:
        # Нов тренинг од нула -- креираме нов SAC модел со сите хиперпараметри
        model = SAC(
            policy="MlpPolicy",             # стандардна мулти-слојна перцептрон политика
            env=train_env,                  # опкружувањето за тренинг
            learning_rate=3e-4,             # стапка на учење
            buffer_size=args.buffer_size,   # капацитет на replay буферот
            batch_size=256,                 # големина на batch за секое градиент-ажурирање
            tau=0.005,                      # коефициент за меко (soft) ажурирање на target мрежите
            gamma=0.99,                     # фактор на попуст (discount) за идни награди
            train_freq=1,                   # тренирај по секој 1 чекор на опкружувањето
            gradient_steps=1,               # 1 градиент-ажурирање по тренинг повик
            learning_starts=10_000,         # почни со учење дури откако ќе се соберат 10,000 транзакции
            ent_coef="auto",                # автоматско прилагодување на коефициентот на ентропија (експлорација)
            use_sde=not args.no_sde,        # користи gSDE освен ако е изречно оневозможено, во некои случаи беше
            sde_sample_freq=4,              # колку често (во чекори) се превзема нов gSDE шум
            policy_kwargs=dict(net_arch=[400, 300]),  # архитектура на невронската мрежа (2 скриени слоја)
            tensorboard_log=os.path.join(args.out_dir, "tb"),  # директориум за TensorBoard логови
            seed=args.seed,                 # семе за репродуцибилност
            verbose=1,                      # ниво на исписи во конзола
        )
        # Нов тренинг -- бројачот на чекори почнува од 0
        reset_num_timesteps = True

    # Callback за постепено воведување (curriculum) на тежините на наградата
    curriculum_cb = CurriculumCallback(
        total_timesteps=args.timesteps,
        base_weights=DEFAULT_WEIGHTS,
        update_every=args.curriculum_update_every,
        verbose=1,
    )

    # Callback за периодична евалуација на моделот и зачувување на најдобриот
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.out_dir,
        log_path=os.path.join(args.out_dir, "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),  # преведено во чекори "по опкружување"
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,   # при евалуација, дејствувај детерминистички (без случаен шум)
        render=False,
    )

    # Callback за периодично зачувување на целосни checkpoint-и (модел + replay буфер)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),  # преведено во чекори "по опкружување"
        save_path=checkpoint_dir,
        name_prefix="ckpt",
        save_replay_buffer=True,
    )

    # Комбинирање на сите callback-и во еден, за да се извршуваат заедно за време на тренингот
    callbacks = CallbackList([curriculum_cb, eval_cb, checkpoint_cb])

    # Главниот повик за тренинг -- ќе трае долго (потенцијално милиони чекори)
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    # По завршување на тренингот, го зачувуваме финалниот модел
    final_path = os.path.join(args.out_dir, "final_model.zip")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")
    print(f"Best model (by eval reward) saved to {os.path.join(args.out_dir, 'best_model.zip')}")

    # Затвораме ги опкружувањата (ослободуваме ресурси/процеси)
    train_env.close()
    eval_env.close()


# Стандардна Python шема -- го извршува main() само ако скриптата се
# повикува директно (не при import)
if __name__ == "__main__":
    main()
