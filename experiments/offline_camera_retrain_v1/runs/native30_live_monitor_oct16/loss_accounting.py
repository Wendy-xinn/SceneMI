"""Display-only accounting of existing logged losses; no training imports."""
import math


def native_contributions(record, observation_coefficients, geometry_coefficients):
    terms = record['terms']
    observation = terms['observation']
    geometry = terms['geometry']
    if set(observation) != set(observation_coefficients):
        raise ValueError('Observation log terms differ from the frozen effective coefficients')
    if set(geometry) != set(geometry_coefficients):
        raise ValueError('Geometry log terms differ from the frozen effective coefficients')
    values = {'loss/total': record['loss'],
              'loss/weighted/base_body_and_gait': terms['base']['total']}
    weighted_observation = {key: observation_coefficients[key] * value
                            for key, value in observation.items()}
    weighted_geometry = {key: geometry_coefficients[key] * value
                         for key, value in geometry.items()}
    values['loss/weighted/observation'] = math.fsum(weighted_observation.values())
    values['loss/weighted/scene_geometry'] = math.fsum(weighted_geometry.values())
    values.update({f'loss/details/weighted_observation/{key}': value
                   for key, value in weighted_observation.items()})
    values.update({f'loss/details/weighted_scene/{key}': value
                   for key, value in weighted_geometry.items()})
    values['loss/audit/weighted_sum_minus_total'] = math.fsum(
        values['loss/weighted/' + key] for key in
        ('base_body_and_gait', 'observation', 'scene_geometry')) - record['loss']
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError('Non-finite logged loss or effective coefficient')
    return values


def legacy_coordination_total(record, contact_coefficient=.1):
    """Actual old55k coordination_v1 objective, for audit only.

    The GT-phase-specific foot_stillness term was removed by this profile;
    phase_free_support_total replaces it. All *_total fields below are already
    weighted. The contact head is added by train.py after supervised_losses.
    """
    terms = record['losses']
    return math.fsum([
        terms['x0'], .2 * terms['rotation_matrix_mse'],
        2. * terms['fk_joint_mse'], 10. * terms['fk_velocity_mse'],
        2. * terms['fk_acceleration_mse'], terms['head_position_mse'],
        .5 * terms['branch_consistency_mse'], terms['gait_total'],
        terms['phase_free_support_total'], terms['coordination_total'],
        contact_coefficient * terms['contact_bce'],
    ])
