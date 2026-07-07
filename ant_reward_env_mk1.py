
# Овозможува користење на нови type-hint синтакси (на пр. `str | None`) и во
# постари верзии на Python, со одложено (lazy) толкување на анотациите
from __future__ import annotations

# NumPy -- за нумерички пресметки (вектори, средни вредности, тригонометрија)
import numpy as np
# Gymnasium -- рамка за RL опкружувања (содржи го стандардното Ant-v5)
#try:
    #import gymnasium as gym
#except ImportError:
import gym  # type: ignore
# deque -- ефикасна структура со фиксна должина, се користи за лизгачки прозорци (history)
from collections import deque


# Распоред на актуатори/qpos за Ant-v5.
# qpos[0:3] = позиција (x,y,z) на торзото, qpos[3:7] = кватернион на торзото, qpos[7:15] = 8 зглобови (колк/глужд).
ACTUATED_QPOS_START = 7  # индекс каде почнуваат зглобовите со актуатори во qpos
ACTUATED_QPOS_END = 15  # exclusive  # индекс каде завршуваат (не се вклучува)
N_ACTUATED = ACTUATED_QPOS_END - ACTUATED_QPOS_START  # 8  # вкупен број на актуирани зглобови (= 8)

# Групи на индекси на актуатори (колк, глужд) за секоја од 4-те нозе на мравката.
# Редоследот одговара на распоредот на актуатори кај Ant-v5: [hip_1,ankle_1,hip_2,ankle_2,hip_3,ankle_3,hip_4,ankle_4]
# нозе = [предна_лева, предна_десна, задна_лева("back_leg"), задна_десна("right_back_leg")]
LEG_GROUPS = [
    (0, 1),  # предна_лева нога (индекси на актуатори)
    (2, 3),  # предна_десна нога
    (4, 5),  # задна_лева нога
    (6, 7),  # задна_десна нога
]

# Имиња на геометриите (geom) на стапалата на секоја нога во стандардниот Ant-v5
# MuJoCo модел, во истиот редослед како LEG_GROUPS погоре.
FOOT_GEOM_NAMES = [
    "left_ankle_geom",     # предна_лева нога
    "right_ankle_geom",    # предна_десна нога
    "third_ankle_geom",    # задна_лева нога
    "fourth_ankle_geom",   # задна_десна нога
]

# Стандардни тежини за секој член од наградата -- може да се override-нат преку
# аргументот `weights` при иницијализација, или динамички преку `set_reward_weights`.
DEFAULT_WEIGHTS = {
    "w_vel": 1.0,          # тежина за брзина во насока на целта (главен поттик за движење)
    "w_survival": 0.02,    # мала константна награда за секој чекор во живот
    "w_avg_vel": 0.5,      # тежина за просечна (изгладена) брзина преку прозорец
    "w_height": 0.5,       # казна за отстапување од целната висина на торзото
    "w_orientation": 0.7,  # казна за накривеност на торзото (roll/pitch грешка)
    "w_yaw": 200.0,        # доминантна казна за ротација околу z-оската (yaw), за движење право
    "w_energy": 0.02,      # казна за потрошена енергија (квадрат на акциите)
    "w_joint_limit": 0.5,  # казна за приближување до граничните вредности на зглобовите
    "w_joint_vel": 0.02,   # казна за прекумерна брзина на зглобовите
    "w_leg_balance": 0.3,  # казна за нерамномерна употреба на 4-те нозе
    "w_smooth": 0.1,       # казна за нагли промени на акцијата (немазност)
    "w_fall": 15.0,        # голема казна доколку мравката падне
    "w_idle": 5.0,         # казна доколку мравката застане/стои неподвижно
}


def _quat_to_euler(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """MuJoCo кватернионите се зачувани како (w, x, y, z). Враќа (roll, pitch, yaw) во радијани."""
    # Расподелба на компонентите на кватернионот
    w, x, y, z = quat_wxyz
    # Пресметка на roll (ротација околу x-оската) преку arctan2 формула од кватернион
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Пресметка на pitch (ротација околу y-оската), со clip за да се избегнат нумерички грешки во arcsin
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Пресметка на yaw (ротација околу z-оската, "хоризонтална" насока на гледање)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    # Враќа ги трите Ојлерови агли
    return roll, pitch, yaw


def _angle_diff(a: float, b: float) -> float:
    """Најмала знаковна разлика меѓу два агли, завиткана во опсегот [-pi, pi]."""
    # Обична разлика меѓу аглите
    d = a - b
    # "Виткање" во опсегот [-pi, pi] за да се избегне скок при премин преку +-pi (пр. 179° vs -179°)
    return (d + np.pi) % (2 * np.pi) - np.pi


class AntCustomRewardEnv(gym.Wrapper):
    """
    Ја обвиткува Ant-v5 и ја заменува нејзината награда со конфигурабилна,
    повеќецелна награда составена од членовите опишани во докстрингот на модулот.

    Параметри
    ----------
    render_mode : str | None
        Се проследува директно во `gym.make('Ant-v5', ...)`.
    target_direction : np.ndarray, shape (2,)
        Единечен (нормализиран) вектор во xy-рамнината кон кој агентот треба да се движи.
    phi_ref : float
        Референтна насока (yaw, во радијани) кон која торзото треба да биде порамнето.
    height_target : float | None
        Посакувана z-висина на торзото. Стандардно е средината на "здравиот" z-опсег на моделот.
    avg_vel_window : int
        Број на чекори за пресметка на изгладениот бонус за просечна брзина.
    upright_threshold : float
        cos(roll)*cos(pitch) под оваа вредност се смета за "превртено" и предизвикува
        паѓање, без разлика на висина/контакт. 1.0 = совршено исправено, 0.0 = превртено
        точно 90 степени. Стандардно 0.4 (~66 степени вкупно накривување) е доста
        толерантно -- намали го за построги/агресивни одови, зголеми за построгост.
    stall_window : int
        Број на последователни чекори за мерење "дали се движи".
    stall_speed_threshold : float
        Под оваа просечна брзина напред (m/s) преку `stall_window` чекори, агентот се
        смета за застоен.
    weights : dict | None
        Почетни тежини на членовите на наградата. Ако не се дадени, се користат DEFAULT_WEIGHTS.
    terminate_on_fall : bool
        Дали детектирано паѓање ја завршува епизодата.
    terminate_on_stall : bool
        Дали детектиран застој ја завршува епизодата. Препорачано е True за време на
        тренинг за да не му се дозволи на агентот да се "скрие" во замрзната состојба
        до крајот на долга епизода; можеш да го поставиш на False за рачна инспекција
        ако сакаш да видиш што се случува после застој.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        target_direction: np.ndarray | tuple[float, float] = (1.0, 0.0),
        phi_ref: float = 0.0,
        height_target: float | None = None,
        avg_vel_window: int = 25,
        upright_threshold: float = 0.4,
        stall_window: int = 100,
        stall_speed_threshold: float = 0.05,
        weights: dict | None = None,
        terminate_on_fall: bool = True,
        terminate_on_stall: bool = True,
    ):
        # Го креираме стандардното Ant-v5 опкружување, но ги гасиме сите вградени
        # награди/казни (forward_reward, ctrl_cost, contact_cost, healthy_reward) и
        # автоматското прекинување при "нездрава" состојба, бидејќи целосно ја
        # заменуваме логиката за награда и прекинување со наша сопствена подолу.
        env = gym.make(
            "Ant-v5",
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,  # ги задржуваме x,y позициите во набљудувањето
            forward_reward_weight=0.0,   # гасиме награда за движење напред
            ctrl_cost_weight=0.0,        # гасиме  казна за акција/контрола
            contact_cost_weight=0.0,     # гасиме  вградената казна за контакт
            healthy_reward=0.0,          # гасиме вградената награда за "здрава" состојба
            terminate_when_unhealthy=False,  #  ја управуваме логиката за паѓање/прекин
        )
        # Го иницијализираме родителскиот gym.Wrapper со креираното опкружување
        super().__init__(env)

        # Ја нормализираме насоката на целта во единечен вектор (доколку нормата не е ~0)
        self.target_direction = np.asarray(target_direction, dtype=np.float64)
        norm = np.linalg.norm(self.target_direction)
        if norm > 1e-8:
            self.target_direction = self.target_direction / norm
        # Референтен yaw агол кон кој треба да е насочено торзото
        self.phi_ref = phi_ref

        # Пристап до MuJoCo моделот за да извадиме идентификатори на тела/геометрии
        model = self.unwrapped.model
        self._torso_id = model.body("torso").id            # ID на телото "торзо"
        self._floor_geom_id = model.geom("floor").id        # ID на геометријата "под"
        self._torso_geom_id = model.geom("torso_geom").id   # ID на геометријата на торзото (за детекција на контакт)
        # ID-иња на геометриите на стапалата на секоја нога (за детекција на контакт со подот)
        self._foot_geom_ids = [model.geom(name).id for name in FOOT_GEOM_NAMES]
        # "Здрав" z-опсег -- ако постои во внатрешното опкружување, го земаме него; инаку стандарден (0.2, 1.0)
        self._healthy_z_range = getattr(self.unwrapped, "_healthy_z_range", (0.2, 1.0))
        # Целна висина: или изречно дадена, или средината на здравиот z-опсег
        self.height_target = (
            height_target
            if height_target is not None
            else 0.5 * (self._healthy_z_range[0] + self._healthy_z_range[1])
        )
        # Опсезите (мин,макс) за секој од 8-те актуирани зглоба (hip_1..ankle_4); го прескокнуваме слободниот зглоб
        self._joint_ranges = model.jnt_range[1:9].copy()  # hip_1..ankle_4, skip free joint

        # Зачувување на параметрите за детекција на паѓање/застој
        self.upright_threshold = upright_threshold
        self.avg_vel_window = avg_vel_window
        self.stall_window = stall_window
        self.stall_speed_threshold = stall_speed_threshold

        # Лизгачки прозорци (со фиксна максимална должина) за брзина кон целта и вкупна брзина
        self._vel_history: deque[float] = deque(maxlen=avg_vel_window)
        self._speed_history: deque[float] = deque(maxlen=stall_window)
        # Лизгачки прозорци на контакт со подот за секоја од 4-те нозе (1.0 = стапалото
        # го допира подот во тој чекор, 0.0 = не го допира)
        self._leg_contact_history: list[deque[float]] = [
            deque(maxlen=stall_window) for _ in range(4)
        ]
        # Претходна акција (за пресметка на "мазност" -- разлика меѓу последователни акции)
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        # Претходна xy позиција (за пресметка на брзина преку конечна разлика)
        self._prev_xy = None

        # Тежини на членовите на наградата -- почнуваме од стандардните, па ги override-ame ако е дадено
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

        # Дали паѓање/застој треба да ја прекинат епизодата
        self.terminate_on_fall = terminate_on_fall
        self.terminate_on_stall = terminate_on_stall

    # ------------------------------------------------------------------
    # Интерфејс за curriculum (постепено менување на тежините за време на тренинг)
    # ------------------------------------------------------------------
    def set_reward_weights(self, weights: dict) -> None:
        """Тежините се фиксни; овој повик намерно не прави ништо."""
        return

    def get_reward_weights(self) -> dict:
        # Враќа копија од тековните тежини (за да не се менуваат случајно однадвор)
        return dict(self.weights)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        # Го ресетираме внатрешното Ant-v5 опкружување
        obs, info = self.env.reset(**kwargs)
        # Празниме ги историите на брзина (нов почеток на епизода)
        self._vel_history.clear()
        self._speed_history.clear()
        for dq in self._leg_contact_history:
            dq.clear()
        # Ресетираме ја претходната акција на нули
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        # Земаме почетна xy позиција за да можеме да пресметаме брзина во првиот чекор
        data = self.unwrapped.data
        self._prev_xy = data.qpos[0:2].copy()
        # Празен речник за компонентите на наградата (ќе се пополни во step)
        info["reward_components"] = {}
        return obs, info

    def step(self, action):
        # Го извршуваме чекорот во внатрешното опкружување; ја игнорираме неговата
        # стандардна награда (_base_reward) и статус на прекин (base_terminated),
        # бидејќи ги пресметуваме сами подолу
        obs, _base_reward, base_terminated, truncated, info = self.env.step(action)

        data = self.unwrapped.data
        w = self.weights

        # Тековна xy позиција и пресметка на брзина преку конечна разлика (позиција/dt)
        xy = data.qpos[0:2].copy()
        dt = self.unwrapped.dt
        velocity_xy = (xy - self._prev_xy) / dt
        self._prev_xy = xy

        # --- Главни поттици (движење) -----------------------------------------------------
        # Проекција на брзината врз насоката на целта (позитивна = движење кон целта)
        r_vel = float(np.dot(velocity_xy, self.target_direction))
        # Вкупен интензитет на брзината (без разлика на насока)
        speed = float(np.linalg.norm(velocity_xy))
        # Додаваме во историите за просек/детекција на застој
        self._vel_history.append(r_vel)
        self._speed_history.append(speed)
        # Просечна (изгладена) брзина кон целта преку прозорецот
        r_avg_vel = float(np.mean(self._vel_history))
        # Константна награда за секој чекор во кој агентот "преживеал"
        r_survival = 1.0

        # --- Ориентација ---------------------------------------------------
        # Кватернион на торзото (w,x,y,z) -> претворен во Ојлерови агли
        quat = data.xquat[self._torso_id].copy()  # (w, x, y, z)
        roll, pitch, yaw = _quat_to_euler(quat)
        # Грешка во yaw во однос на референтната насока (завиткана во [-pi, pi])
        yaw_err = _angle_diff(yaw, self.phi_ref)
        # Казна за ротација околу z-оската (yaw), одвоена и доминантна за да се
        # спречи вртење и се задржи право движење
        p_yaw = yaw_err**2
        # Казна за накривеност на торзото (roll/pitch)
        p_orientation = roll**2 + pitch**2
        # "Исправеност": 1.0 = совршено исправен, <=0 = превртен зад хоризонтала
        uprightness = np.cos(roll) * np.cos(pitch)  # 1.0 = upright, <=0 = tipped past horizontal

        # --- Висина ---------------------------------------------------------
        # Тековна z-висина на торзото
        z = data.qpos[2]
        # Казна = квадратно отстапување од целната висина
        p_height = (z - self.height_target) ** 2

        # --- Енергија (среден квадрат на применетата акција/момент) --------------------
        p_energy = float(np.mean(np.square(action)))

        # --- Гранични вредности и брзини на зглобовите ---------------------------------
        # Тековни позиции и брзини на 8-те актуирани зглоба
        joint_qpos = data.qpos[ACTUATED_QPOS_START:ACTUATED_QPOS_END]
        joint_qvel = data.qvel[6:14]  # 6 free-joint dof, then 8 actuated dof
        # Долна и горна граница за секој зглоб
        lo = self._joint_ranges[:, 0]
        hi = self._joint_ranges[:, 1]
        # Опсег на секој зглоб (со минимум 1e-6 за да се избегне делење со нула)
        span = np.clip(hi - lo, 1e-6, None)
        # Нормализирано растојание до долната и до горната граница (0..1)
        dist_to_lo = (joint_qpos - lo) / span
        dist_to_hi = (hi - joint_qpos) / span
        # "Блискост" до најблиската граница: 0 = во средина, 1 = точно на границата
        closeness = 1.0 - np.clip(np.minimum(dist_to_lo, dist_to_hi), 0.0, 1.0)
        # Казна за приближување до граничните вредности (среден квадрат на closeness)
        p_joint_limit = float(np.mean(closeness**2))
        # Казна за брзина на зглобовите (среден квадрат на аглеските брзини)
        p_joint_vel = float(np.mean(np.square(joint_qvel)))

        # --- Баланс на нозете: казнува нерамномерна употреба на 4-те нозе, без да
        # наметнува конкретен временски/координационен образец меѓу нив (за разлика
        # од старата симетрија-трот, ова не поддржува кучешки одбор -- само
        # обесхрабрува занемарување на било која нога, на пр. задните нозе).
        # Употреба = дел од времето (во лизгачки прозорец) во кој стапалото на
        # секоја нога реално го допира подот; казна = варијанса на употребата
        # меѓу 4-те нозе. ------------------
        foot_touch = self._foot_contacts()
        for leg_idx in range(4):
            self._leg_contact_history[leg_idx].append(foot_touch[leg_idx])
        leg_usage = np.array([
            float(np.mean(dq)) if len(dq) > 0 else 0.0
            for dq in self._leg_contact_history
        ])
        p_leg_balance = float(np.var(leg_usage))

        # --- Мазност (среден квадрат на промената на акцијата) -------------------------
        p_smooth = float(np.mean(np.square(action - self._prev_action)))
        # Ја зачувуваме тековната акција како "претходна" за следниот чекор
        self._prev_action = np.asarray(action, dtype=np.float64).copy()

        # --- Детекција на паѓање: висина ВОН опсег, експлицитен контакт торзо/под,
        # ИЛИ колапс на исправеноста (го фаќа случајот "превртен, но технички во
        # опсег на висина", кој порано се пропуштал). --------------
        fell_by_height = not (self._healthy_z_range[0] <= z <= self._healthy_z_range[1])
        fell_by_contact = self._torso_touching_floor()
        fell_by_tipping = uprightness < self.upright_threshold
        # Паѓање = ако важи барем еден од трите услови
        fell = fell_by_height or fell_by_contact or fell_by_tipping
        p_fall = 1.0 if fell else 0.0

        # --- Детекција на застој: продолжена речиси-нулта брзина. Се проверува само
        # откако ќе имаме целосно пополнет прозорец, за да не се активира во првите
        # неколку чекори од епизодата. -------------------------------
        stalled = (
            len(self._speed_history) == self.stall_window
            and float(np.mean(self._speed_history)) < self.stall_speed_threshold
        )
        p_idle = 1.0 if stalled else 0.0

        # Вкупна награда = линеарна комбинација на сите поттици (позитивни) минус
        # сите казни (негативни), секоја помножена со сопствената тежина
        reward = (
            w["w_vel"] * r_vel
            + w["w_survival"] * r_survival
            + w["w_avg_vel"] * r_avg_vel
            - w["w_height"] * p_height
            - w["w_yaw"] * p_yaw
            - w["w_orientation"] * p_orientation
            - w["w_energy"] * p_energy
            - w["w_joint_limit"] * p_joint_limit
            - w["w_joint_vel"] * p_joint_vel
            - w["w_leg_balance"] * p_leg_balance
            - w["w_smooth"] * p_smooth
            - w["w_fall"] * p_fall
            - w["w_idle"] * p_idle
        )

        # Епизодата завршува ако: внатрешното опкружување го побарало тоа, ИЛИ имало
        # паѓање (и е овозможено прекинување при паѓање), ИЛИ имало застој (и е
        # овозможено прекинување при застој)
        terminated = bool(
            base_terminated
            or (fell and self.terminate_on_fall)
            or (stalled and self.terminate_on_stall)
        )

        # Ги запишуваме сите компоненти на наградата и корисни дијагностички
        # вредности во `info`, за анализа/логирање (пр. TensorBoard, евалуација)
        info["reward_components"] = {
            "r_vel": r_vel,
            "r_survival": r_survival,
            "r_avg_vel": r_avg_vel,
            "p_height": p_height,
            "p_yaw": p_yaw,
            "p_orientation": p_orientation,
            "p_energy": p_energy,
            "p_joint_limit": p_joint_limit,
            "p_joint_vel": p_joint_vel,
            "p_leg_balance": p_leg_balance,
            "p_smooth": p_smooth,
            "p_fall": p_fall,
            "p_idle": p_idle,
            "total_reward": float(reward),
            "yaw": yaw,
            "roll": roll,
            "pitch": pitch,
            "z_height": float(z),
            "uprightness": float(uprightness),
            "speed": speed,
            "leg_usage_front_left": float(leg_usage[0]),
            "leg_usage_front_right": float(leg_usage[1]),
            "leg_usage_back_left": float(leg_usage[2]),
            "leg_usage_back_right": float(leg_usage[3]),
        }

        # Враќа: набљудување, скаларна награда, дали е завршена епизодата (terminated),
        # дали е прекината поради лимит на чекори (truncated), и дополнителни инфо
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _foot_contacts(self) -> np.ndarray:
        # Помошна функција: за секоја од 4-те нозе, проверува дали нејзиното
        # стапало и подот се меѓу активните MuJoCo контакти во тековниот чекор
        data = self.unwrapped.data
        touching = np.zeros(4, dtype=np.float64)
        for i in range(data.ncon):
            c = data.contact[i]
            geoms = (c.geom1, c.geom2)
            if self._floor_geom_id not in geoms:
                continue
            for leg_idx, foot_id in enumerate(self._foot_geom_ids):
                if foot_id in geoms:
                    touching[leg_idx] = 1.0
        return touching

    def _torso_touching_floor(self) -> bool:
        # Помошна функција: проверува дали торзото и подот се меѓу активните MuJoCo контакти
        data = self.unwrapped.data
        # Поминуваме низ сите тековни контакти во симулацијата
        for i in range(data.ncon):
            c = data.contact[i]
            geoms = (c.geom1, c.geom2)
            # Ако и торзото и подот се дел од истиот контакт -> торзото допира под
            if self._torso_geom_id in geoms and self._floor_geom_id in geoms:
                return True
        # Немаше таков контакт
        return False


def make_ant_env(render_mode: str | None = None, **kwargs) -> AntCustomRewardEnv:
    """Фабричка функција што ја користат train.py / evaluation notebook-от."""
    # Едноставно ја креира и враќа обвиената околина со проследените параметри
    return AntCustomRewardEnv(render_mode=render_mode, **kwargs)
