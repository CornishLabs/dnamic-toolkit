import numpy as np
import pytest

from dnamic_toolkit.physics.alkali_polarizability import (
    AlkaliState,
    dynamic_polarizability_au,
    dynamic_polarizability_from_components,
    dynamic_polarizability_uncertainty_au,
    interpolate_portal_polarizability,
    load_portal_polarizability_csv,
    load_portal_polarizability_folder,
    portal_states,
    portal_table,
    tensor_prefactor,
)


def test_state_labels_parse_portal_and_fraction_forms():
    assert AlkaliState.from_label("Rb", "5p3").label == "5p3/2"
    assert AlkaliState.from_label("rubidium", "5p3/2").display_label == "Rb 5p3/2"
    assert AlkaliState.from_label("caesium", "6s1/2", isotope=133).display_label == (
        "Cs133 6s1/2"
    )


def test_portal_csv_loader_and_interpolation(tmp_path):
    csv_path = tmp_path / "Rb1_5p3.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Rb1,5p3/2,,,",
                "Wavelength,alpha_0,     Uncertainty,alpha_2,     Uncertainty",
                "600.0,10.0,1.0,4.0,0.4",
                "602.0,14.0,2.0,8.0,0.8",
            ]
        )
    )

    table = load_portal_polarizability_csv(csv_path)
    components = interpolate_portal_polarizability(table, [601.0])

    assert table.state.species == "Rb"
    assert table.ion_stage == 1
    assert table.state.label == "5p3/2"
    assert components.alpha0_au == pytest.approx([12.0])
    assert components.alpha1_au == pytest.approx([0.0])
    assert components.alpha2_au == pytest.approx([6.0])
    assert components.alpha0_uncertainty_au == pytest.approx([1.5])
    assert components.alpha2_uncertainty_au == pytest.approx([0.6])


def test_portal_folder_loader_returns_tables_by_species_and_state(tmp_path):
    csv_path = tmp_path / "Rb1_5s.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Rb1,5s1/2,,,",
                "Wavelength,alpha_0,     Uncertainty,alpha_2,     Uncertainty",
                "600.0,10.0,1.0,0.0,0.0",
            ]
        )
    )

    tables = load_portal_polarizability_folder(tmp_path)

    assert portal_table(tables, "Rb", "5s1").alpha0_au == pytest.approx([10.0])
    assert [state.label for state in portal_states(tables, "Rb")] == ["5s1/2"]


def test_dynamic_polarizability_combines_scalar_vector_and_tensor_terms():
    result = dynamic_polarizability_au(
        alpha0_au=np.array([100.0]),
        alpha1_au=np.array([10.0]),
        alpha2_au=np.array([20.0]),
        j=1.5,
        m_j=1.5,
        circularity=1.0,
    )

    assert result == pytest.approx([130.0])
    assert tensor_prefactor(j=1.5, m_j=1.5) == pytest.approx(1.0)
    assert tensor_prefactor(j=0.5, m_j=0.5) == pytest.approx(0.0)


def test_dynamic_helpers_work_from_portal_components(tmp_path):
    csv_path = tmp_path / "Rb1_5p3.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Rb1,5p3/2,,,",
                "Wavelength,alpha_0,     Uncertainty,alpha_2,     Uncertainty",
                "600.0,10.0,1.0,4.0,0.4",
            ]
        )
    )
    table = load_portal_polarizability_csv(csv_path)
    components = interpolate_portal_polarizability(table, [600.0])

    assert dynamic_polarizability_from_components(components) == pytest.approx([14.0])
    assert dynamic_polarizability_uncertainty_au(
        components.alpha0_uncertainty_au,
        components.alpha2_uncertainty_au,
        j=components.state.j,
        m_j=components.state.j,
    ) == pytest.approx([np.sqrt(1.0**2 + 0.4**2)])

