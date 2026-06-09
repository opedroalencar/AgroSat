"""Componentes de UI puros: leem templates HTML/CSS do disco e devolvem strings
prontas pra st.markdown(..., unsafe_allow_html=True). Sem chamadas Streamlit aqui."""

from pathlib import Path

_DIR = Path(__file__).parent
_TPL = _DIR / "templates"
_CSS = _DIR / "styles.css"


def _render(nome_tpl: str, **vars) -> str:
    """Carrega template do disco e substitui placeholders {nome} por valores."""
    return (_TPL / nome_tpl).read_text(encoding="utf-8").format(**vars)


def load_css() -> str:
    """Retorna o bloco <style>...</style> com todo o CSS do dashboard."""
    return f"<style>{_CSS.read_text(encoding='utf-8')}</style>"


def sidebar_subtitle(texto: str) -> str:
    return _render("sidebar_subtitle.html", texto=texto)


def sidebar_info(titulo: str, texto: str) -> str:
    return _render("sidebar_info.html", titulo=titulo, texto=texto)


def sidebar_footer(href: str, link_label: str, linha2: str) -> str:
    return _render("sidebar_footer.html", href=href, link_label=link_label, linha2=linha2)


def hero(title: str, subtitle: str, pill: str) -> str:
    return _render("hero.html", title=title, subtitle=subtitle, pill=pill)


def banner_critico(nomes: str, icon: str = "🚨") -> str:
    return _render("banner_critico.html", nomes=nomes, icon=icon)


def kpi_card(label: str, value: str, unit: str = "", sub: str = "", variant: str = "") -> str:
    return _render(
        "kpi_card.html",
        label=label, value=value, unit=unit, sub=sub, variant=variant,
    )


def kpi_grid(cards: list) -> str:
    return _render("kpi_grid.html", items="".join(cards))


def isa_legend(itens: list) -> str:
    """itens: lista de tuplas (categoria, cor_hex, faixa_str)."""
    parts = [
        _render("isa_legend_item.html", cat=cat, cor=cor, faixa=faixa)
        for cat, cor, faixa in itens
    ]
    return _render("isa_legend.html", items="".join(parts))


def region_card_isa(cat: str, cor: str, css_cat: str,
                    chuva_mm: str, dias_secos: str, temp_c: str) -> str:
    return _render(
        "region_card_isa.html",
        cat=cat, cor=cor, css_cat=css_cat,
        chuva_mm=chuva_mm, dias_secos=dias_secos, temp_c=temp_c,
    )


def region_card_risco(icon: str, nome: str, nivel: str, css: str,
                      chuva_mm: str, temp_c: str, dias_secos: str) -> str:
    return _render(
        "region_card_risco.html",
        icon=icon, nome=nome, nivel=nivel, css=css,
        chuva_mm=chuva_mm, temp_c=temp_c, dias_secos=dias_secos,
    )


def region_card_ml(icon: str, nome: str, pred_label: str, css: str,
                   confianca: str, chuva_mm: str, isa: str, isa_cat: str,
                   dias_secos: str, pct_secos: str) -> str:
    return _render(
        "region_card_ml.html",
        icon=icon, nome=nome, pred_label=pred_label, css=css,
        confianca=confianca, chuva_mm=chuva_mm, isa=isa, isa_cat=isa_cat,
        dias_secos=dias_secos, pct_secos=pct_secos,
    )
