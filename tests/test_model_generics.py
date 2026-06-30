import torch
import pytest

from utils.model_generics import GVP, EquivariantData, EquivariantLayerNorm

NUM_TEST_NODES = 30
NUM_TEST_SCALARS = 128
NUM_TEST_VECTORS = 5


@pytest.fixture
def gvp_layer() -> GVP:
    return GVP((0, NUM_TEST_VECTORS), (0, NUM_TEST_VECTORS))


@pytest.fixture
def zeros_scalars_and_input_vectors() -> EquivariantData:
    return EquivariantData(torch.zeros(NUM_TEST_NODES, NUM_TEST_SCALARS), torch.rand((NUM_TEST_NODES, NUM_TEST_VECTORS, 3)))


@pytest.fixture
def non_zero_scalars_and_input_vectors() -> EquivariantData:
    return EquivariantData(torch.rand(NUM_TEST_NODES, NUM_TEST_SCALARS), torch.rand((NUM_TEST_NODES, NUM_TEST_VECTORS, 3)))


def test_gvp_layer_zeros_scalar_input_does_not_modify_input_scalars(zeros_scalars_and_input_vectors: EquivariantData, gvp_layer: GVP):
    output = gvp_layer(zeros_scalars_and_input_vectors)
    assert torch.isclose(output.scalars, zeros_scalars_and_input_vectors.scalars).all().item(), "GVP layer should not modify input scalars when layer is created with no scalar input even when called on input with scalars"


def test_gvp_layer_nonzero_scalar_input_does_not_modify_input_scalars(non_zero_scalars_and_input_vectors: EquivariantData, gvp_layer: GVP):
    output = gvp_layer(non_zero_scalars_and_input_vectors)
    assert torch.isclose(output.scalars, non_zero_scalars_and_input_vectors.scalars).all().item(), "GVP layer should not modify input scalars when layer is created with no scalar input even when called on input with scalars"


def test_equivariant_norm_selectively_ignores_scalars(non_zero_scalars_and_input_vectors: EquivariantData):
    complete_norm_layer = EquivariantLayerNorm((NUM_TEST_SCALARS, NUM_TEST_VECTORS), vector_only=False)
    vector_only_norm_layer = EquivariantLayerNorm((NUM_TEST_SCALARS, NUM_TEST_VECTORS), vector_only=True)

    assert (vector_only_norm_layer(non_zero_scalars_and_input_vectors).scalars == non_zero_scalars_and_input_vectors.scalars).all().item(), "Vector only norm layer should not modify input scalars"
    assert (complete_norm_layer(non_zero_scalars_and_input_vectors).scalars != non_zero_scalars_and_input_vectors.scalars).any().item(), "Complete norm layer should modify input scalars"