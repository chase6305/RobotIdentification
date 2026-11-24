"""Collect named joint samples from contact-free MuJoCo arm models."""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

MAX_STEP_SECONDS = 0.001
WARMUP_SECONDS = 3.0
POSITION_GAIN_S2 = 10000.0
VELOCITY_GAIN_S = 200.0


@dataclass(frozen=True)
class ArmDynamics:
    model: mujoco.MjModel
    positions: np.ndarray
    dofs: np.ndarray

    @classmethod
    def from_urdf(cls, urdf, joints, drive_parameters=None):
        if drive_parameters is None:
            model = mujoco.MjModel.from_xml_path(str(urdf))
        else:
            spec = mujoco.MjSpec.from_file(str(urdf))
            for name, (armature, damping) in zip(joints,
                                                 drive_parameters,
                                                 strict=True):
                joint = spec.joint(name)
                if joint is None:
                    raise ValueError(f"expected hinge joint {name} in {urdf}")
                joint.armature = armature
                joint.damping[0] = damping
            model = spec.compile()
        model.opt.gravity[:] = (0.0, 0.0, -9.81)
        return cls._named_model(model, joints, urdf)

    @classmethod
    def from_mjb(cls, filename, joints):
        return cls._named_model(mujoco.MjModel.from_binary_path(str(filename)),
                                joints, filename)

    @classmethod
    def from_mjcf(cls, filename, joints):
        return cls._named_model(mujoco.MjModel.from_xml_path(str(filename)),
                                joints, filename)

    @classmethod
    def _named_model(cls, model, joints, filename):
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joints
        ]
        if (model.nq != len(joints) or model.nv != len(joints) or
                len(set(joint_ids)) != len(joints) or
                any(index < 0 for index in joint_ids) or np.any(
                    model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError(
                f"expected exactly the requested hinge joints in {filename}")
        return cls(model, model.jnt_qposadr[joint_ids],
                   model.jnt_dofadr[joint_ids])

    def inverse_torque(self, data, position, velocity, acceleration):
        data.qpos[self.positions] = position
        data.qvel[self.dofs] = velocity
        data.qacc[self.dofs] = acceleration
        mujoco.mj_inverse(self.model, data)
        return data.qfrc_inverse[self.dofs]


def save_contact_free_mjcf(model, filename):

    def vector(values):
        return " ".join(map(str, values))

    xml = ET.Element("mujoco", model=filename.stem)
    ET.SubElement(xml, "compiler", angle="radian", inertiafromgeom="false")
    option = ET.SubElement(xml,
                           "option",
                           timestep=str(model.opt.timestep),
                           gravity=vector(model.opt.gravity),
                           integrator="Euler")
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP):
        ET.SubElement(option, "flag", eulerdamp="disable")
    bodies = [ET.SubElement(xml, "worldbody")]
    for index in range(1, model.nbody):
        body = ET.SubElement(bodies[model.body_parentid[index]],
                             "body",
                             name=model.body(index).name,
                             pos=vector(model.body_pos[index]),
                             quat=vector(model.body_quat[index]))
        bodies.append(body)
        if model.body_mass[index] > 0.0:
            ET.SubElement(body,
                          "inertial",
                          mass=str(model.body_mass[index]),
                          pos=vector(model.body_ipos[index]),
                          quat=vector(model.body_iquat[index]),
                          diaginertia=vector(model.body_inertia[index]))
        first = model.body_jntadr[index]
        for joint in range(first, first + model.body_jntnum[index]):
            dof = model.jnt_dofadr[joint]
            element = ET.SubElement(body,
                                    "joint",
                                    name=model.joint(joint).name,
                                    type="hinge",
                                    pos=vector(model.jnt_pos[joint]),
                                    axis=vector(model.jnt_axis[joint]),
                                    armature=str(model.dof_armature[dof]),
                                    damping=str(model.dof_damping[dof]))
            if model.jnt_limited[joint]:
                element.set("range", vector(model.jnt_range[joint]))
            if model.jnt_actfrclimited[joint]:
                element.set("actuatorfrcrange",
                            vector(model.jnt_actfrcrange[joint]))
    # Preserve fitted double-precision values when round-tripping to text.
    ET.indent(xml, space="  ")
    ET.ElementTree(xml).write(filename, encoding="utf-8", xml_declaration=True)


def inverse_dynamics(plant, motions, effort_limits=None):
    data = mujoco.MjData(plant.model)
    dof = len(plant.dofs)
    result = {}
    for name, motion in motions.items():
        labels = motion.copy()
        for row in labels:
            position, velocity, acceleration, _ = np.split(row[1:], 4)
            torque = plant.inverse_torque(data, position, velocity,
                                          acceleration)
            if data.nefc:
                raise ValueError(
                    f"active MuJoCo constraints in {name} at t={row[0]:.6g} s")
            row[-dof:] = torque
        if not np.isfinite(labels[:, -dof:]).all():
            raise ValueError(
                f"{name} inverse dynamics produced nonfinite torque")
        if effort_limits is not None and np.any(
                np.abs(labels[:, -dof:]) > effort_limits):
            raise ValueError(
                f"{name} inverse dynamics exceeded a URDF joint effort limit")
        result[name] = labels
    return result


def configure_integrator(model, step_seconds):
    model.opt.timestep = step_seconds
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)


def replay_windows(plant, dense, window_count, window_seconds):
    """Re-integrate a reloaded model under the recorded holdout torques."""
    time = dense[:, 0]
    last_start = int(np.searchsorted(time, time[-1] - window_seconds)) - 1
    if (window_seconds <= 0.0 or last_start < 0 or
            not 1 <= window_count <= last_start + 1):
        raise ValueError(
            "dense holdout cannot provide the requested replay windows")
    positions, velocities, _, torques = np.split(dense[:, 1:], 4, axis=1)
    dt = time[1] - time[0]
    if not np.allclose(np.diff(time), dt, rtol=0.0, atol=1e-12):
        raise ValueError("MuJoCo replay requires a uniform integration grid")
    model = plant.model
    configure_integrator(model, dt)
    limits = model.jnt_range[model.dof_jntid[plant.dofs]]
    data = mujoco.MjData(model)
    complete = 0
    squared_error = 0.0
    terms = 0
    maximum_final_error = 0.0
    for window in range(window_count):
        start = window * last_start // max(1, window_count - 1)
        end = int(np.searchsorted(time, time[start] + window_seconds))
        mujoco.mj_resetData(model, data)
        data.qpos[plant.positions] = positions[start]
        data.qvel[plant.dofs] = velocities[start]
        window_error = 0.0
        for index in range(start, end):
            data.qfrc_applied[plant.dofs] = torques[index]
            mujoco.mj_step(model, data)
            position = data.qpos[plant.positions]
            if (data.nefc or data.warning.number.any() or
                    not np.isfinite(position).all() or
                    np.any(position < limits[:, 0]) or
                    np.any(position > limits[:, 1])):
                break
            error = position - positions[index + 1]
            window_error += np.dot(error, error)
        else:
            complete += 1
            squared_error += window_error
            terms += (end - start) * positions.shape[1]
            maximum_final_error = max(maximum_final_error,
                                      float(np.max(np.abs(error))))
    all_complete = complete == window_count
    return {
        "window_seconds":
            window_seconds,
        "windows":
            window_count,
        "complete_windows":
            complete,
        "position_rmse_rad":
            math.sqrt(squared_error / terms) if all_complete else None,
        "maximum_final_position_error_rad":
            maximum_final_error if all_complete else None,
    }


def harmonic_states(coefficients, time):
    offset = np.asarray(coefficients["offset_rad"])
    sine = np.asarray(coefficients["sin_rad"])
    cosine = np.asarray(coefficients["cos_rad"])
    omega = 2.0 * np.pi * coefficients["frequency_hz"] * np.arange(
        1, sine.shape[1] + 1)
    angle = time[:, np.newaxis] * omega
    sin_angle, cos_angle = np.sin(angle), np.cos(angle)
    position = offset + sin_angle @ sine.T + cos_angle @ cosine.T
    velocity = cos_angle @ (omega * sine).T - sin_angle @ (omega * cosine).T
    acceleration = -sin_angle @ (omega**2 * sine).T - cos_angle @ (omega**2 *
                                                                   cosine).T
    return position, velocity, acceleration


def rollout(plant,
            controller,
            motions,
            references,
            effort_limits,
            jitter_seconds=0.0,
            seed=0,
            replay=False,
            torque_delay_seconds=0.0):
    labels = {}
    diagnostics = {}
    dense_holdout = None
    for name, motion in motions.items():
        noise_seed = np.random.SeedSequence([seed, 2, int(name == "holdout")])
        labels[name], diagnostics[name], dense = _rollout_motion(
            plant, controller, motion, references[name], effort_limits, name,
            jitter_seconds, noise_seed, replay and name == "holdout",
            0.0 if name == "holdout" else torque_delay_seconds)
        if dense is not None:
            dense_holdout = dense
    return labels, diagnostics, dense_holdout


def _rollout_motion(plant,
                    controller,
                    motion,
                    reference,
                    effort_limits,
                    name,
                    jitter_seconds,
                    noise_seed,
                    save_dense,
                    torque_delay_seconds=0.0):
    sample_seconds = motion[:, 0]
    duration = sample_seconds[-1]
    interval = duration / (len(motion) - 1)
    if not np.allclose(
            sample_seconds, np.arange(len(motion)) * interval, rtol=0.0,
            atol=1e-12):
        raise ValueError(
            f"{name} requires uniformly sampled reference states starting at zero"
        )
    substeps = math.ceil(interval / MAX_STEP_SECONDS)
    step_count = substeps * (len(motion) - 1)
    step_seconds = duration / step_count
    if jitter_seconds >= 0.5 * interval:
        raise ValueError(
            f"{name} timestamp jitter must be below half the sampling interval")
    if torque_delay_seconds and abs(torque_delay_seconds) >= min(
            0.5 * duration, WARMUP_SECONDS):
        raise ValueError(
            f"{name} torque delay exceeds the valid capture interval")
    nominal_steps = np.arange(len(motion)) * substeps
    jitter_steps = math.floor(jitter_seconds / step_seconds)
    generator = np.random.default_rng(noise_seed)
    sample_steps = nominal_steps.copy()
    sample_steps[1:-1] += generator.integers(-jitter_steps,
                                             jitter_steps + 1,
                                             size=len(motion) - 2)
    measured_times = sample_seconds + step_seconds * (sample_steps -
                                                      nominal_steps)
    warmup_steps = math.ceil(WARMUP_SECONDS / step_seconds)
    final_step = step_count + max(
        0, math.ceil(-torque_delay_seconds / step_seconds))
    ticks = np.arange(-warmup_steps, final_step + 1)
    reference_states = harmonic_states(reference, ticks * step_seconds)
    label_positions, label_velocities, label_accelerations, label_torques = np.split(
        np.empty_like(motion[:, 1:]), 4, axis=1)
    dense = np.empty((step_count + 1, motion.shape[1])) if save_dense else None
    torque_history = np.empty(
        (len(ticks), len(effort_limits))) if torque_delay_seconds else None

    configure_integrator(plant.model, step_seconds)
    plant_data = mujoco.MjData(plant.model)
    control_data = mujoco.MjData(controller.model)
    plant_data.qpos[plant.positions] = reference_states[0][0]
    plant_data.qvel[plant.dofs] = reference_states[1][0]
    peak_command = np.zeros_like(effort_limits)
    negative_effort_limits = -effort_limits
    saturation_steps = 0
    row = 0
    for index, (q_ref, dq_ref, ddq_ref) in enumerate(zip(*reference_states)):
        step = index - warmup_steps
        position = plant_data.qpos[plant.positions]
        velocity = plant_data.qvel[plant.dofs]
        desired_acceleration = (ddq_ref + POSITION_GAIN_S2 *
                                (q_ref - position) + VELOCITY_GAIN_S *
                                (dq_ref - velocity))
        command = controller.inverse_torque(control_data, position, velocity,
                                            desired_acceleration)
        time = step * step_seconds
        if not np.isfinite(command).all():
            raise ValueError(
                f"{name} computed non-finite torque at t={time:.6g} s")
        if control_data.nefc:
            raise ValueError(
                f"{name} tracking reached a joint constraint at t={time:.6g} s")
        if abs(plant_data.time - index * step_seconds) > 1e-9:
            raise ValueError(
                f"{name} simulation time became invalid at t={time:.6g} s")
        magnitude = np.abs(command)
        np.maximum(peak_command, magnitude, out=peak_command)
        torque = command
        if (magnitude > effort_limits).any():
            torque = np.clip(command, negative_effort_limits, effort_limits)
            saturation_steps += 1
        if torque_history is not None:
            torque_history[index] = torque
        plant_data.qfrc_applied[plant.dofs] = torque
        if step < final_step:
            mujoco.mj_step(plant.model, plant_data)
        else:
            mujoco.mj_forward(plant.model, plant_data)
        if plant_data.nefc or plant_data.warning.number.any():
            raise ValueError(
                f"{name} MuJoCo constraint or integration warning at t={time:.6g} s"
            )
        if dense is not None and 0 <= step <= step_count:
            dense[step, 0] = time
            dense[step, 1:] = np.concatenate(
                (position, velocity, plant_data.qacc[plant.dofs], torque))
        if row < len(sample_steps) and step == sample_steps[row]:
            label_positions[row] = position
            label_velocities[row] = velocity
            # During Euler stepping qacc still describes the pre-update state.
            label_accelerations[row] = plant_data.qacc[plant.dofs]
            label_torques[row] = torque
            row += 1
    if torque_history is not None:
        torque_indices = sample_steps + warmup_steps - torque_delay_seconds / step_seconds
        all_indices = np.arange(len(ticks))
        for joint in range(len(effort_limits)):
            label_torques[:, joint] = np.interp(torque_indices, all_indices,
                                                torque_history[:, joint])
    labels = np.column_stack((measured_times, label_positions, label_velocities,
                              label_accelerations, label_torques))
    if row != len(motion):
        raise ValueError(f"{name} did not capture every requested sample")
    if not np.isfinite(labels).all():
        raise ValueError(f"{name} generated non-finite joint samples")
    tracking_error = label_positions - reference_states[0][sample_steps +
                                                           warmup_steps]
    diagnostics = {
        "simulation_step_s":
            step_seconds,
        "integration_steps":
            step_count,
        "warmup_s":
            warmup_steps * step_seconds,
        "max_timestamp_jitter_s":
            float(np.max(np.abs(measured_times - sample_seconds))),
        "torque_channel_delay_s":
            torque_delay_seconds,
        "saturation_steps":
            saturation_steps,
        "peak_effort_fraction":
            float(
                (np.minimum(peak_command, effort_limits) / effort_limits).max()
            ),
        "position_tracking_rmse_rad":
            float(np.sqrt(np.mean(tracking_error**2))),
        "joint_position_tracking_rmse_rad":
            np.sqrt(np.mean(tracking_error**2, axis=0)).tolist(),
    }
    return labels, diagnostics, dense
