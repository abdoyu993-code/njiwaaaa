/* =====================================================================
   Plateforme NARSA — logique de l'interface
   ===================================================================== */

(function () {
  "use strict";

  /* ------------------------------ Utilitaires ------------------------ */

  const API_BASE =
    window.location.protocol === "file:"
      ? "http://127.0.0.1:8000"
      : window.location.origin;

  const $ = (selector, root) =>
    (root || document).querySelector(selector);
  const $$ = (selector, root) =>
    Array.from((root || document).querySelectorAll(selector));

  const SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
  ];

  const DOMAIN_LABELS = {
    permis: "Permis de conduire",
    vehicule: "Véhicule et carte grise",
    visite_technique: "Visite technique",
    infractions: "Infractions et amendes",
    accident: "Accidents",
    prevention: "Prévention routière",
    contact_reclamation: "Contact et réclamation",
    conversation: "Conversation",
    inconnu: "Non déterminé",
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function domainLabel(domain) {
    if (!domain) return "Non déterminé";
    return (
      DOMAIN_LABELS[domain] ||
      domain.replace(/_/g, " ")
    );
  }

  function formatDate(value, withTime) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "—";

    const options = withTime
      ? {
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        }
      : { day: "2-digit", month: "2-digit", year: "numeric" };

    return date.toLocaleString("fr-FR", options);
  }

  function formatDuration(milliseconds) {
    const value = Number(milliseconds) || 0;
    if (value < 1000) return `${Math.round(value)} ms`;
    return `${(value / 1000).toFixed(2)} s`;
  }

  async function apiRequest(path, options) {
    const config = Object.assign(
      { headers: {} },
      options || {}
    );

    if (config.body && typeof config.body !== "string") {
      config.body = JSON.stringify(config.body);
      config.headers["Content-Type"] = "application/json";
    }

    const response = await fetch(API_BASE + path, config);
    const text = await response.text();
    let payload = null;

    try {
      payload = text ? JSON.parse(text) : null;
    } catch (error) {
      payload = null;
    }

    if (!response.ok) {
      const detail =
        (payload && payload.detail) ||
        `Erreur ${response.status}`;

      throw new Error(
        typeof detail === "string"
          ? detail
          : "Requête refusée par le serveur."
      );
    }

    return payload;
  }

  /* --------------------- Rendu texte type markdown ------------------- */

  function renderRichText(raw) {
    const text = String(raw == null ? "" : raw).trim();
    if (!text) return "";

    const html = [];
    let listType = null;
    let listItems = [];
    let paragraph = [];

    function flushList() {
      if (!listItems.length) return;

      html.push(
        "<" +
          listType +
          ">" +
          listItems.map((item) => "<li>" + item + "</li>").join("") +
          "</" +
          listType +
          ">"
      );

      listItems = [];
      listType = null;
    }

    function flushParagraph() {
      if (!paragraph.length) return;

      html.push("<p>" + paragraph.join("<br />") + "</p>");
      paragraph = [];
    }

    text.split("\n").forEach((rawLine) => {
      const line = rawLine.trim();

      if (!line) {
        flushParagraph();
        flushList();
        return;
      }

      // Intertitre Markdown : rendu comme un libellé.
      if (/^#{1,6}\s+/.test(line)) {
        flushParagraph();
        flushList();
        html.push(
          "<p><strong>" +
            inlineFormat(line.replace(/^#{1,6}\s+/, "")) +
            "</strong></p>"
        );
        return;
      }

      if (/^[-•*]\s+/.test(line)) {
        flushParagraph();

        if (listType !== "ul") {
          flushList();
          listType = "ul";
        }

        listItems.push(
          inlineFormat(line.replace(/^[-•*]\s+/, ""))
        );
        return;
      }

      if (/^\d+[.)]\s+/.test(line)) {
        flushParagraph();

        if (listType !== "ol") {
          flushList();
          listType = "ol";
        }

        listItems.push(
          inlineFormat(line.replace(/^\d+[.)]\s+/, ""))
        );
        return;
      }

      flushList();
      paragraph.push(inlineFormat(line));
    });

    flushParagraph();
    flushList();

    return html.join("");
  }

  function inlineFormat(line) {
    return escapeHtml(line)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(
        /(https?:\/\/[^\s<]+[^\s<.,;:)\]])/g,
        '<a href="$1" target="_blank" rel="noopener">$1</a>'
      );
  }

  /* ------------------------------- Routeur --------------------------- */

  const ROUTES = {
    accueil: "page-accueil",
    parcours: "page-parcours",
    assistant: "page-assistant",
    reclamation: "page-reclamation",
    suivi: "page-suivi",
    admin: "page-admin",
    dashboard: "page-dashboard",
  };

  function currentRoute() {
    const hash = window.location.hash.replace(/^#\/?/, "");
    const name = hash.split("?")[0] || "accueil";
    return ROUTES[name] ? name : "accueil";
  }

  function navigate(route) {
    window.location.hash = "#/" + route;
  }

  function applyAuthState() {
    const connected = Boolean(Admin.token());

    document.body.classList.toggle("is-agent", connected);

    if (connected) {
      $("#topbar-agent-name").textContent =
        sessionStorage.getItem("narsa-admin-name") || "Agent";
    }
  }

  function showRoute() {
    applyAuthState();

    let route = currentRoute();

    if (route === "dashboard" && !Admin.token()) {
      route = "admin";
      window.location.hash = "#/admin";
    }

    if (route === "admin" && Admin.token()) {
      route = "dashboard";
      window.location.hash = "#/dashboard";
    }

    document.body.dataset.route = route;

    Object.keys(ROUTES).forEach((name) => {
      const section = document.getElementById(ROUTES[name]);
      if (section) section.hidden = name !== route;
    });

    $$("[data-nav]").forEach((link) => {
      link.classList.toggle(
        "is-active",
        link.dataset.nav === route
      );
    });

    $("#topbar").classList.remove("is-open");
    window.scrollTo({ top: 0, behavior: "instant" });

    if (route === "reclamation") Complaints.ensureCategories();
    if (route === "dashboard") Admin.refresh();
    if (route === "assistant") Chat.focus();
    if (route === "parcours") Journey.show();
  }

  /* --------------------------- Lecteur de source --------------------- */

  const SourceViewer = {
    init() {
      $$("[data-close-modal]").forEach((node) => {
        node.addEventListener("click", () => this.close());
      });

      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") this.close();
      });
    },

    close() {
      $("#source-modal").hidden = true;
    },

    async open(path, title) {
      const modal = $("#source-modal");
      const body = $("#source-body");

      $("#source-title").textContent =
        title || "Document officiel";
      body.innerHTML =
        '<p class="muted">Chargement du document...</p>';
      modal.hidden = false;

      try {
        const data = await apiRequest(
          "/documents/content?path=" +
            encodeURIComponent(path)
        );

        $("#source-title").textContent = data.title;
        body.innerHTML =
          '<p class="muted">' +
          escapeHtml(data.relative_path) +
          "</p>" +
          renderRichText(data.content);
      } catch (error) {
        body.innerHTML =
          '<p class="alert">' +
          escapeHtml(error.message) +
          "</p>";
      }
    },
  };

  function confidenceLabel(data) {
    const score = Number(data.confidence || 0);
    const sources = (data.sources || []).length;

    if (data.status === "no_context_found" || !sources) {
      return {
        level: "faible",
        label: "Informations insuffisantes",
      };
    }

    if (score >= 0.75 && sources >= 1) {
      return {
        level: "forte",
        label: "Réponse fortement étayée",
      };
    }

    return {
      level: "partielle",
      label: "Sources partielles",
    };
  }

  /* ------------------------------ Parcours --------------------------- */

  const Journey = {
    current: null,

    init() {
      $("#problem-form").addEventListener(
        "submit",
        (event) => {
          event.preventDefault();
          this.plan($("#problem-input").value.trim());
        }
      );

      $$("[data-goal]").forEach((button) => {
        button.addEventListener("click", () => {
          $("#problem-input").value = button.dataset.goal;
          this.plan(button.dataset.goal);
        });
      });

      $("#journey-resume").addEventListener(
        "submit",
        (event) => {
          event.preventDefault();
          const reference = $(
            "#journey-reference"
          ).value.trim();

          if (reference) this.load(reference);
        }
      );
    },

    reference() {
      return localStorage.getItem("narsa-journey");
    },

    async plan(goal) {
      const hint = $("#problem-hint");

      if (goal.length < 5) {
        hint.textContent =
          "Décrivez votre situation en quelques mots.";
        return;
      }

      hint.textContent = "Analyse de votre situation...";

      try {
        const data = await apiRequest("/journeys/plan", {
          method: "POST",
          body: {
            goal: goal,
            session_id: Chat.sessionId,
          },
        });

        if (!data.detected) {
          hint.textContent = data.message;
          navigate("parcours");
          this.current = null;
          this.render();
          return;
        }

        hint.textContent = "";
        $("#problem-input").value = "";
        this.current = data.journey;
        localStorage.setItem(
          "narsa-journey",
          data.journey.reference
        );
        navigate("parcours");
        this.render();
      } catch (error) {
        hint.textContent = error.message;
      }
    },

    async startFromCatalogue(key) {
      try {
        const data = await apiRequest(
          "/journeys/start/" + encodeURIComponent(key),
          {
            method: "POST",
            body: {
              goal: "Parcours choisi dans le catalogue",
              session_id: Chat.sessionId,
            },
          }
        );

        this.current = data;
        localStorage.setItem(
          "narsa-journey",
          data.reference
        );
        this.render();
      } catch (error) {
        window.alert(error.message);
      }
    },

    async load(reference) {
      try {
        const data = await apiRequest(
          "/journeys/" + encodeURIComponent(reference)
        );

        this.current = data;
        localStorage.setItem(
          "narsa-journey",
          data.reference
        );
      } catch (error) {
        this.current = null;
        localStorage.removeItem("narsa-journey");
      }

      this.render();
    },

    show() {
      if (this.current) {
        this.render();
        return;
      }

      const saved = this.reference();

      if (saved) {
        this.load(saved);
        return;
      }

      this.render();
    },

    async toggle(index, done) {
      if (!this.current) return;

      try {
        const data = await apiRequest(
          "/journeys/" +
            encodeURIComponent(this.current.reference) +
            "/steps/" +
            index,
          {
            method: "POST",
            body: { done: done },
          }
        );

        this.current = data;
        this.render();
      } catch (error) {
        window.alert(error.message);
      }
    },

    async renderCatalogue() {
      const container = $("#journey-catalogue");

      if (container.dataset.loaded) return;

      try {
        const data = await apiRequest("/journeys/catalogue");

        container.innerHTML = data.journeys
          .map(
            (item) =>
              '<button class="chip" type="button" data-journey="' +
              escapeHtml(item.journey_key) +
              '">' +
              escapeHtml(item.title) +
              " · " +
              item.total_steps +
              " étapes</button>"
          )
          .join("");

        container.dataset.loaded = "1";

        $$("[data-journey]", container).forEach((button) => {
          button.addEventListener("click", () => {
            this.startFromCatalogue(button.dataset.journey);
          });
        });
      } catch (error) {
        container.innerHTML =
          '<p class="muted">Catalogue indisponible.</p>';
      }
    },

    render() {
      const container = $("#journey-container");
      const empty = $("#journey-empty");

      if (!this.current) {
        container.innerHTML = "";
        empty.hidden = false;
        this.renderCatalogue();
        return;
      }

      empty.hidden = true;

      const journey = this.current;

      container.innerHTML =
        '<div class="card">' +
        '<div class="journey__head">' +
        "<div><h2>" +
        escapeHtml(journey.title) +
        "</h2>" +
        '<p class="muted">' +
        escapeHtml(journey.summary) +
        "</p></div>" +
        '<span class="journey__reference">' +
        escapeHtml(journey.reference) +
        "</span></div>" +
        '<p class="journey__goal">« ' +
        escapeHtml(journey.goal) +
        " »</p>" +
        '<div class="progress">' +
        '<span class="progress__track"><span class="progress__fill" style="width:' +
        journey.progress +
        '%"></span></span>' +
        '<span class="progress__label">' +
        journey.completed_count +
        "/" +
        journey.total_steps +
        " étapes terminées</span></div>" +
        '<div class="journey-meta">' +
        (journey.deadline
          ? '<div class="journey-meta__item journey-meta__item--deadline">' +
            '<svg class="ico"><use href="#i-history"/></svg>' +
            "<span><strong>Délai :</strong> " +
            escapeHtml(journey.deadline) +
            "</span></div>"
          : "") +
        (journey.warning
          ? '<div class="journey-meta__item journey-meta__item--warning">' +
            '<svg class="ico"><use href="#i-shield"/></svg>' +
            "<span>" +
            escapeHtml(journey.warning) +
            "</span></div>"
          : "") +
        "</div>" +
        '<ol class="steps">' +
        journey.steps
          .map((step) => this.renderStep(step, journey))
          .join("") +
        "</ol>" +
        '<p class="muted">Conservez la référence ' +
        escapeHtml(journey.reference) +
        " pour retrouver ce parcours plus tard.</p>" +
        "</div>";

      $$("[data-step-toggle]", container).forEach((button) => {
        button.addEventListener("click", () => {
          this.toggle(
            Number(button.dataset.stepToggle),
            button.dataset.stepDone !== "1"
          );
        });
      });

      $$("[data-source]", container).forEach((button) => {
        button.addEventListener("click", () => {
          SourceViewer.open(
            button.dataset.source,
            button.dataset.sourceTitle
          );
        });
      });
    },

    renderStep(step, journey) {
      const isNext =
        journey.next_step &&
        journey.next_step.index === step.index;

      return (
        '<li class="step' +
        (step.done ? " step--done" : "") +
        (isNext ? " step--next" : "") +
        '">' +
        '<div class="step__head">' +
        '<span class="step__title">' +
        escapeHtml(step.title) +
        "</span>" +
        (step.duration
          ? '<span class="step__duration">' +
            escapeHtml(step.duration) +
            "</span>"
          : "") +
        "</div>" +
        '<p class="step__detail">' +
        escapeHtml(step.detail) +
        "</p>" +
        (step.documents.length
          ? '<p class="step__docs-title">Documents à préparer</p>' +
            '<ul class="step__docs">' +
            step.documents
              .map(
                (document) =>
                  "<li>" + escapeHtml(document) + "</li>"
              )
              .join("") +
            "</ul>"
          : "") +
        '<div class="step__actions">' +
        '<button class="btn btn--sm step__toggle ' +
        (step.done ? "btn--ghost" : "btn--primary") +
        '" type="button" data-step-toggle="' +
        step.index +
        '" data-step-done="' +
        (step.done ? "1" : "0") +
        '">' +
        (step.done
          ? "Marquer à refaire"
          : "Marquer comme terminée") +
        "</button>" +
        step.forms
          .map(
            (form) =>
              '<a class="attachment" href="' +
              escapeHtml(API_BASE + form.download_url) +
              '" target="_blank" rel="noopener">' +
              '<svg class="ico"><use href="#i-download"/></svg>' +
              escapeHtml(form.title) +
              "</a>"
          )
          .join("") +
        step.links
          .map(
            (link) =>
              '<a class="attachment" href="' +
              escapeHtml(link.url) +
              '" target="_blank" rel="noopener">' +
              '<svg class="ico"><use href="#i-link"/></svg>' +
              escapeHtml(link.label) +
              "</a>"
          )
          .join("") +
        (step.source
          ? '<button class="link-source" type="button" data-source="' +
            escapeHtml(step.source) +
            '" data-source-title="Document officiel">' +
            "Voir le document officiel</button>"
          : "") +
        "</div></li>"
      );
    },
  };

  /* -------------------------------- Chat ----------------------------- */

  const Chat = {
    sessionId: null,
    context: null,
    busy: false,

    init() {
      this.messages = $("#chat-messages");
      this.input = $("#chat-input");

      this.sessionId =
        localStorage.getItem("narsa-session-id") ||
        "web-" +
          Math.random().toString(36).slice(2, 10) +
          Date.now().toString(36);
      localStorage.setItem(
        "narsa-session-id",
        this.sessionId
      );

      $("#chat-form").addEventListener("submit", (event) => {
        event.preventDefault();
        const value = this.input.value.trim();
        if (value) this.send(value);
      });

      $$("[data-ask]").forEach((button) => {
        button.addEventListener("click", () => {
          const question = button.dataset.ask;
          navigate("assistant");
          this.send(question);
        });
      });

      const heroSearch = $("#hero-search");

      if (heroSearch) {
        heroSearch.addEventListener("submit", (event) => {
          event.preventDefault();
          const field = $("#hero-search-input");
          const value = field.value.trim();
          if (!value) return;
          field.value = "";
          navigate("assistant");
          this.send(value);
        });
      }

      this.appendAssistant({
        answer:
          "Bonjour 👋 Je suis l'assistant intelligent de la NARSA. " +
          "Posez-moi une question sur le permis, la carte grise, la " +
          "visite technique, les infractions ou la prévention routière.",
        status: "local_response",
      });
    },

    focus() {
      window.setTimeout(() => {
        if (this.input) this.input.focus();
      }, 60);
    },

    scrollToEnd() {
      this.messages.scrollTop = this.messages.scrollHeight;
    },

    appendUser(question) {
      const node = document.createElement("div");
      node.className = "msg msg--user";
      node.innerHTML =
        '<span class="msg__avatar"><svg class="ico"><use href="#i-user"/></svg></span>' +
        '<div class="msg__bubble">' +
        renderRichText(question) +
        '<span class="msg__time">' +
        new Date().toLocaleTimeString("fr-FR", {
          hour: "2-digit",
          minute: "2-digit",
        }) +
        "</span></div>";

      this.messages.appendChild(node);
      this.scrollToEnd();
    },

    appendTyping() {
      const node = document.createElement("div");
      node.className = "msg msg--bot";
      node.id = "typing-indicator";
      node.innerHTML =
        '<span class="msg__avatar"><svg class="ico"><use href="#i-chat"/></svg></span>' +
        '<div class="msg__bubble"><span class="typing">' +
        "<span></span><span></span><span></span></span></div>";

      this.messages.appendChild(node);
      this.scrollToEnd();
      return node;
    },

    appendAssistant(data) {
      const node = document.createElement("div");
      node.className = "msg msg--bot";

      let html =
        '<span class="msg__avatar"><svg class="ico"><use href="#i-chat"/></svg></span>' +
        '<div class="msg__bubble">' +
        renderRichText(data.answer);

      const forms = (data.forms || []).filter(
        (form) => form.download_url
      );
      const links = (data.service_links || []).filter(
        (link) => link.url && link.label
      );

      if (forms.length || links.length) {
        html += '<div class="msg__attachments">';

        forms.forEach((form) => {
          const url = form.is_external
            ? form.download_url
            : API_BASE + form.download_url;

          html +=
            '<a class="attachment" href="' +
            escapeHtml(url) +
            '" target="_blank" rel="noopener">' +
            '<svg class="ico"><use href="#i-download"/></svg>' +
            escapeHtml(form.title || form.filename) +
            "</a>";
        });

        links.forEach((link) => {
          html +=
            '<a class="attachment" href="' +
            escapeHtml(link.url) +
            '" target="_blank" rel="noopener">' +
            '<svg class="ico"><use href="#i-link"/></svg>' +
            escapeHtml(link.label) +
            "</a>";
        });

        html += "</div>";
      }


      // Niveau de confiance en clair + accès aux sources.
      if (data.status && data.status !== "local_response") {
        const verdict = confidenceLabel(data);
        const sources = (data.sources || []).slice(0, 3);
        const meta = data.metadata || {};
        const interactionId = meta.interaction_id;

        html +=
          '<div class="msg__footer">' +
          // Le niveau « faible » n'est pas affiché au citoyen :
          // un avertissement rouge sous une réponse utile est
          // anxiogène et dévalorise l'assistant. Le niveau reste
          // calculé et disponible dans « Détails techniques ».
          (verdict.level === "faible"
            ? ""
            : '<span class="confidence confidence--' +
              verdict.level +
              '">' +
              verdict.label +
              "</span>") +
          sources
            .map(
              (source, index) =>
                '<button class="link-source" type="button" data-source="' +
                escapeHtml(source.relative_path || "") +
                '" data-source-title="' +
                escapeHtml(source.title || "Document") +
                '">[Source ' +
                (index + 1) +
                "] " +
                escapeHtml(
                  source.title || source.filename || "document"
                ) +
                "</button>"
            )
            .join("") +
          '<button class="link-source" type="button" data-tech>' +
          "Détails techniques</button>" +
          "</div>";

        // Avis du citoyen, rattaché à l'interaction journalisée.
        if (interactionId) {
          html +=
            '<div class="feedback" data-feedback-block>' +
            "<span>Cette réponse vous a-t-elle été utile ?</span>" +
            '<button class="feedback__btn" type="button" data-feedback="1" data-interaction="' +
            interactionId +
            '" aria-label="Réponse utile">👍 Oui</button>' +
            '<button class="feedback__btn" type="button" data-feedback="0" data-interaction="' +
            interactionId +
            '" aria-label="Réponse pas utile">👎 Non</button>' +
            "</div>";
        }

        // Bloc technique, replié : réservé au debug et aux
        // analytics, jamais affiché d'emblée au citoyen.
        html +=
          '<div class="msg__meta" hidden>' +
          "<span><strong>Domaine :</strong> " +
          escapeHtml(domainLabel(data.domain)) +
          " — <strong>Intention :</strong> " +
          escapeHtml(data.intent || "—") +
          " (" +
          Number(data.confidence || 0).toFixed(2) +
          ")</span>" +
          "<span><strong>Statut :</strong> " +
          escapeHtml(data.status || "—") +
          (data.generated
            ? " — génération " + escapeHtml(data.model || "")
            : meta.extractive
              ? " — réponse extractive"
              : "") +
          (meta.cached ? " — servi par le cache" : "") +
          "</span>" +
          "<span><strong>Langue détectée :</strong> " +
          escapeHtml(meta.language || "fr") +
          "</span>" +
          "<span><strong>Temps :</strong> " +
          formatDuration(meta.total_time_ms) +
          " (classification " +
          formatDuration(meta.classification_time_ms) +
          ", recherche " +
          formatDuration(meta.retrieval_time_ms) +
          ", génération " +
          formatDuration(meta.generation_time_ms) +
          ")</span>" +
          (sources.length
            ? "<span><strong>Sources :</strong> " +
              escapeHtml(
                sources
                  .map(
                    (source) =>
                      (source.title ||
                        source.filename ||
                        "source") +
                      " (" +
                      Number(
                        source.best_similarity_score || 0
                      ).toFixed(2) +
                      ")"
                  )
                  .join(", ")
              ) +
              "</span>"
            : "") +
          "</div>";
      }

      const related = (data.related_questions || []).slice(
        0,
        3
      );

      if (related.length) {
        html +=
          '<div class="related">' +
          '<span class="related__label">Questions liées</span>' +
          related
            .map(
              (question) =>
                '<button class="chip" type="button" data-related="' +
                escapeHtml(question) +
                '">' +
                escapeHtml(question) +
                "</button>"
            )
            .join("") +
          "</div>";
      }

      html +=
        '<span class="msg__time">' +
        new Date().toLocaleTimeString("fr-FR", {
          hour: "2-digit",
          minute: "2-digit",
        }) +
        "</span></div>";

      node.innerHTML = html;

      $$("[data-related]", node).forEach((button) => {
        button.addEventListener("click", () => {
          this.send(button.dataset.related);
        });
      });

      $$("[data-source]", node).forEach((button) => {
        button.addEventListener("click", () => {
          if (!button.dataset.source) return;

          SourceViewer.open(
            button.dataset.source,
            button.dataset.sourceTitle
          );
        });
      });

      const techButton = $("[data-tech]", node);
      const techBlock = $(".msg__meta", node);

      if (techButton && techBlock) {
        techButton.addEventListener("click", () => {
          techBlock.hidden = !techBlock.hidden;
          techButton.textContent = techBlock.hidden
            ? "Détails techniques"
            : "Masquer les détails";
        });
      }

      $$("[data-feedback]", node).forEach((button) => {
        button.addEventListener("click", () => {
          this.sendFeedback(
            Number(button.dataset.interaction),
            button.dataset.feedback === "1",
            button.closest("[data-feedback-block]")
          );
        });
      });

      this.messages.appendChild(node);
      this.scrollToEnd();
    },

    async sendFeedback(interactionId, helpful, block) {
      if (!interactionId || !block) return;

      block.innerHTML =
        '<span class="feedback__thanks">Merci pour votre retour.</span>';

      try {
        await apiRequest("/feedback", {
          method: "POST",
          body: {
            interaction_id: interactionId,
            helpful: helpful,
          },
        });
      } catch (error) {
        block.innerHTML =
          '<span class="feedback__thanks">Retour non enregistré : ' +
          escapeHtml(error.message) +
          "</span>";
      }
    },

    async send(question) {
      if (this.busy) return;

      this.busy = true;
      this.input.value = "";
      $("#chat-send").disabled = true;
      this.appendUser(question);

      const typing = this.appendTyping();

      try {
        const payload = {
          question: question,
          top_k: 3,
          session_id: this.sessionId,
        };

        if (this.context) payload.context = this.context;

        const data = await apiRequest("/ask", {
          method: "POST",
          body: payload,
        });

        typing.remove();
        this.appendAssistant(data);

        if (
          data.status !== "local_response" &&
          data.domain &&
          data.intent
        ) {
          this.context = {
            previous_question: data.question,
            previous_domain: data.domain,
            previous_intent: data.intent,
          };
        }
      } catch (error) {
        typing.remove();
        this.appendAssistant({
          answer:
            "Le service est momentanément indisponible : " +
            error.message +
            "\n\nVérifiez que l'API NARSA est démarrée, puis réessayez.",
          status: "error",
        });
      } finally {
        this.busy = false;
        $("#chat-send").disabled = false;
        this.focus();
      }
    },
  };

  /* --------------------------- Réclamations -------------------------- */

  const Complaints = {
    categoriesLoaded: false,
    analyzed: false,

    init() {
      $("#complaint-form").addEventListener(
        "submit",
        (event) => {
          event.preventDefault();
          this.submit(event.target);
        }
      );

      $("#complaint-new").addEventListener("click", () => {
        $("#complaint-success").hidden = true;
        $("#complaint-form").hidden = false;
        $("#complaint-form").reset();
        $("#complaint-analysis").hidden = true;
        this.analyzed = false;
      });

      $("#tracking-form").addEventListener(
        "submit",
        (event) => {
          event.preventDefault();
          this.track(event.target);
        }
      );

      $("#analyze-button").addEventListener("click", () => {
        this.analyze();
      });

      // Analyse automatique dès que la description est
      // suffisamment détaillée, une seule fois.
      const message = $(
        "#complaint-form textarea[name=message]"
      );

      message.addEventListener("blur", () => {
        if (
          !this.analyzed &&
          message.value.trim().length >= 40
        ) {
          this.analyze();
        }
      });
    },

    async ensureCategories() {
      if (this.categoriesLoaded) return;

      const select = $("#complaint-category");

      try {
        const data = await apiRequest("/complaints/categories");

        select.innerHTML =
          '<option value="">Sélectionnez une catégorie</option>' +
          data.categories
            .map(
              (item) =>
                '<option value="' +
                escapeHtml(item.value) +
                '">' +
                escapeHtml(item.label) +
                "</option>"
            )
            .join("");

        this.categoriesLoaded = true;
      } catch (error) {
        select.innerHTML =
          '<option value="">Catégories indisponibles</option>';
      }
    },

    async analyze() {
      const form = $("#complaint-form");
      const message = form.message.value.trim();
      const panel = $("#complaint-analysis");
      const button = $("#analyze-button");

      if (message.length < 10) {
        panel.hidden = false;
        panel.innerHTML =
          '<p class="muted">Décrivez votre demande en quelques ' +
          "phrases pour lancer l'analyse.</p>";
        return;
      }

      this.analyzed = true;
      button.disabled = true;
      button.textContent = "Analyse en cours...";
      panel.hidden = false;
      panel.innerHTML =
        '<p class="muted">Analyse de votre demande...</p>';

      try {
        const data = await apiRequest("/complaints/analyze", {
          method: "POST",
          body: {
            message: message,
            subject: form.subject.value.trim() || null,
          },
        });

        if (
          data.category_is_reliable &&
          !form.category.value
        ) {
          form.category.value = data.category;
        }

        panel.innerHTML = this.renderAnalysis(data);

        const dismiss = $("#analysis-dismiss", panel);

        if (dismiss) {
          dismiss.addEventListener("click", () => {
            panel.hidden = true;
            this.analyzed = false;
            form.reset();
            window.location.hash = "#/accueil";
          });
        }
      } catch (error) {
        panel.innerHTML =
          '<p class="muted">Analyse indisponible : ' +
          escapeHtml(error.message) +
          "</p>";
      } finally {
        button.disabled = false;
        button.innerHTML =
          '<svg class="ico"><use href="#i-chat"/></svg>' +
          "Analyser ma demande";
      }
    },

    renderAnalysis(data) {
      const answer = data.instant_answer;

      let html =
        "<h4>Analyse de votre demande</h4>" +
        '<div class="analysis__row">' +
        "<span><strong>Catégorie proposée :</strong> " +
        escapeHtml(data.category_label) +
        (data.category_is_reliable
          ? " (confiance " +
            Number(data.category_confidence).toFixed(2) +
            ")"
          : " — à confirmer") +
        "</span>" +
        priorityBadge(data.priority, data.priority_label) +
        "</div>";

      if ((data.priority_reasons || []).length) {
        html +=
          '<ul class="analysis__reasons">' +
          data.priority_reasons
            .map(
              (reason) =>
                "<li>" + escapeHtml(reason) + "</li>"
            )
            .join("") +
          "</ul>";
      }

      if (answer) {
        html +=
          '<div class="analysis__answer">' +
          "<h5>Une réponse officielle existe déjà pour ce sujet</h5>" +
          renderRichText(answer.answer) +
          (answer.sources.length
            ? '<p class="muted">Source(s) : ' +
              escapeHtml(answer.sources.join(", ")) +
              "</p>"
            : "") +
          '<div class="analysis__actions">' +
          '<button class="btn btn--ghost btn--sm" type="button" id="analysis-dismiss">' +
          "Cette réponse me suffit</button>" +
          '<span class="muted">Sinon, complétez et envoyez votre réclamation.</span>' +
          "</div></div>";
      } else {
        html +=
          '<p class="muted">Aucune réponse documentaire immédiate ' +
          "sur ce sujet : votre réclamation sera transmise à un agent.</p>";
      }

      return html;
    },

    async submit(form) {
      const errorBox = $("#complaint-error");
      const button = $("#complaint-submit");

      errorBox.hidden = true;
      button.disabled = true;
      button.textContent = "Envoi en cours...";

      const payload = {};
      new FormData(form).forEach((value, key) => {
        const text = String(value).trim();
        if (text) payload[key] = text;
      });

      try {
        const data = await apiRequest("/complaints", {
          method: "POST",
          body: payload,
        });

        $("#complaint-reference").textContent = data.reference;
        $("#complaint-success").hidden = false;
        form.hidden = true;
        form.reset();

        $("#complaint-track-link").addEventListener(
          "click",
          () => {
            $("#tracking-reference").value = data.reference;
          },
          { once: true }
        );
      } catch (error) {
        errorBox.textContent = error.message;
        errorBox.hidden = false;
      } finally {
        button.disabled = false;
        button.textContent = "Suivant";
      }
    },

    async track(form) {
      const errorBox = $("#tracking-error");
      const result = $("#tracking-result");

      $("#page-suivi").classList.remove("has-result");

      errorBox.hidden = true;
      result.innerHTML = "";

      const payload = {};
      new FormData(form).forEach((value, key) => {
        payload[key] = String(value).trim();
      });

      try {
        const data = await apiRequest("/complaints/track", {
          method: "POST",
          body: payload,
        });

        result.innerHTML = this.renderDetail(data);
        $("#page-suivi").classList.add("has-result");

        const printButton = $("#track-print", result);

        if (printButton) {
          printButton.addEventListener("click", () =>
            window.print()
          );
        }

        $$("[data-copy]", result).forEach((button) => {
          button.addEventListener("click", () => {
            navigator.clipboard
              ?.writeText(button.dataset.copy)
              .catch(() => {});
          });
        });
      } catch (error) {
        errorBox.textContent = error.message;
        errorBox.hidden = false;
      }
    },

    renderDetail(data) {
      const detail = (icon, label, value, extra) =>
        '<div class="detail-item">' +
        '<span class="detail-item__icon"><svg class="ico"><use href="#i-' +
        icon +
        '"/></svg></span>' +
        "<div><small>" +
        escapeHtml(label) +
        "</small><strong>" +
        escapeHtml(value) +
        "</strong>" +
        (extra || "") +
        "</div></div>";

      return (
        '<div class="card">' +
        '<div class="detail-head">' +
        "<h3>Détails de votre réclamation</h3>" +
        statusBadge(data.status, data.status_label) +
        "</div>" +
        '<div class="detail-grid">' +
        detail(
          "tag",
          "Référence",
          data.reference,
          '<button class="copy-btn" type="button" data-copy="' +
            escapeHtml(data.reference) +
            '" title="Copier la référence">' +
            '<svg class="ico"><use href="#i-copy"/></svg></button>'
        ) +
        detail("user", "Déposant", data.full_name) +
        detail("category", "Catégorie", data.category_label) +
        detail(
          "calendar",
          "Date de dépôt",
          formatDate(data.created_at, true)
        ) +
        "</div>" +
        '<div class="summary-block">' +
        "<h4>Résumé de la réclamation</h4>" +
        "<p>" +
        escapeHtml(data.subject) +
        " — " +
        escapeHtml(data.message) +
        "</p>" +
        (data.official_response
          ? '<div class="official-response">' +
            '<h4><svg class="ico"><use href="#i-mailsend"/></svg>' +
            "Réponse officielle de la NARSA</h4>" +
            renderRichText(data.official_response) +
            '<p class="official-response__meta">Transmise le ' +
            formatDate(data.responded_at, true) +
            (data.responded_by
              ? " par " + escapeHtml(data.responded_by)
              : "") +
            "</p></div>"
          : data.admin_note
            ? '<p class="alert alert--ok"><strong>Note de la NARSA :</strong> ' +
              escapeHtml(data.admin_note) +
              "</p>"
            // Aucune réponse encore rédigée : on l'annonce
            // explicitement plutôt que de laisser un vide.
            : '<div class="response-pending">' +
              '<h4><svg class="ico"><use href="#i-spinner"/></svg>' +
              "Réponse officielle en attente</h4>" +
              "<p>Aucune réponse officielle n'a encore été rédigée " +
              "pour ce dossier. Vous serez prévenu par e-mail dès " +
              "qu'un agent y aura répondu, et la réponse complète " +
              "s'affichera ici.</p></div>") +
        "</div>" +
        '<div class="timeline-block">' +
        "<h4>Parcours de votre réclamation</h4>" +
        this.renderTimeline(data) +
        "</div>" +
        '<div class="track-actions">' +
        '<button class="btn btn--ghost" type="button" id="track-print">' +
        '<svg class="ico"><use href="#i-download"/></svg>' +
        "Télécharger l'historique (PDF)</button>" +
        "</div></div>"
      );
    },

    renderTimeline(data) {
      // Quatre étapes fixes, alimentées par les événements
      // réellement enregistrés sur le dossier.
      const events = data.events || [];
      const dateFor = (status) => {
        const event = events.find(
          (item) => item.status === status
        );
        return event ? event.created_at : null;
      };

      const rejected = data.status === "rejetee";

      const stages = [
        {
          icon: "check",
          title: "Réclamation reçue",
          text: "Votre réclamation a été enregistrée avec succès.",
          date: dateFor("recue") || data.created_at,
        },
        {
          icon: "spinner",
          title: "En cours de traitement",
          text: "Votre dossier est en cours d'étude par nos équipes.",
          date: dateFor("en_cours"),
        },
        {
          icon: "mailsend",
          title: "Réponse envoyée",
          text: "Vous recevrez une réponse officielle par e-mail et dans votre espace.",
          date: dateFor("traitee") || dateFor("rejetee"),
        },
        {
          icon: "archive",
          title: rejected ? "Dossier clôturé" : "Clôturée",
          text: "Votre réclamation sera clôturée après votre retour.",
          date: dateFor("traitee") || dateFor("rejetee"),
        },
      ];

      const order = ["recue", "en_cours", "traitee", "rejetee"];
      let reached = order.indexOf(data.status);

      if (reached < 0) reached = 0;
      if (data.status === "traitee" || rejected) reached = 2;

      return (
        '<ul class="track-timeline">' +
        stages
          .map((stage, index) => {
            const done = index < reached;
            const current = index === reached;
            const shown = done || current ? stage.date : null;

            return (
              '<li class="track-step' +
              (done ? " track-step--done" : "") +
              (current ? " track-step--current" : "") +
              '">' +
              '<span class="track-step__icon"><svg class="ico"><use href="#i-' +
              stage.icon +
              '"/></svg></span>' +
              '<div class="track-step__body">' +
              "<strong>" +
              escapeHtml(stage.title) +
              "</strong><p>" +
              escapeHtml(stage.text) +
              "</p>" +
              (current
                ? '<span class="step-badge">Étape actuelle</span>'
                : "") +
              "</div>" +
              '<span class="track-step__meta">' +
              (shown ? formatDate(shown, true) : "—") +
              "</span></li>"
            );
          })
          .join("") +
        "</ul>"
      );
    },
  };

  function statusBadge(status, label) {
    return (
      '<span class="badge badge--' +
      escapeHtml(status) +
      '">' +
      escapeHtml(label || status) +
      "</span>"
    );
  }

  function priorityBadge(priority, label) {
    const value = priority || "normale";

    return (
      '<span class="badge badge--' +
      escapeHtml(value) +
      '">' +
      escapeHtml(label || value) +
      "</span>"
    );
  }

  /* ------------------------------ Graphiques ------------------------- */

  const Charts = {
    tooltip: null,

    ensureTooltip(container) {
      let tooltip = container.querySelector(".tooltip");

      if (!tooltip) {
        tooltip = document.createElement("div");
        tooltip.className = "tooltip";
        container.appendChild(tooltip);
      }

      return tooltip;
    },

    line(container, series) {
      container.innerHTML = "";

      if (!series.length) {
        container.innerHTML =
          '<p class="muted">Aucune donnée sur la période.</p>';
        return;
      }

      const width = 560;
      const height = 220;
      const padding = { top: 16, right: 18, bottom: 30, left: 40 };
      const innerWidth = width - padding.left - padding.right;
      const innerHeight = height - padding.top - padding.bottom;

      const values = series.map((point) => point.count);
      const maxValue = Math.max(4, ...values);
      const stepX =
        series.length > 1
          ? innerWidth / (series.length - 1)
          : 0;

      const pointX = (index) => padding.left + index * stepX;
      const pointY = (value) =>
        padding.top +
        innerHeight -
        (value / maxValue) * innerHeight;

      const linePath = series
        .map(
          (point, index) =>
            (index ? "L" : "M") +
            pointX(index).toFixed(1) +
            " " +
            pointY(point.count).toFixed(1)
        )
        .join(" ");

      const areaPath =
        linePath +
        " L" +
        pointX(series.length - 1).toFixed(1) +
        " " +
        (padding.top + innerHeight) +
        " L" +
        pointX(0).toFixed(1) +
        " " +
        (padding.top + innerHeight) +
        " Z";

      const ticks = [0, 0.5, 1].map((ratio) => {
        const value = Math.round(maxValue * ratio);
        return {
          value: value,
          y: pointY(value),
        };
      });

      const svg = [
        '<svg viewBox="0 0 ' +
          width +
          " " +
          height +
          '" role="img" aria-label="Évolution des interactions par jour">',
        '<defs><linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">' +
          '<stop offset="0%" stop-color="#2a78d6" stop-opacity=".22"/>' +
          '<stop offset="100%" stop-color="#2a78d6" stop-opacity="0"/>' +
          "</linearGradient></defs>",
      ];

      ticks.forEach((tick) => {
        svg.push(
          '<line x1="' +
            padding.left +
            '" x2="' +
            (width - padding.right) +
            '" y1="' +
            tick.y.toFixed(1) +
            '" y2="' +
            tick.y.toFixed(1) +
            '" stroke="#e8edf5" stroke-width="1"/>' +
            '<text x="' +
            (padding.left - 8) +
            '" y="' +
            (tick.y + 4).toFixed(1) +
            '" text-anchor="end" font-size="11" fill="#64748b">' +
            tick.value +
            "</text>"
        );
      });

      svg.push(
        '<path d="' + areaPath + '" fill="url(#areaGrad)"/>'
      );
      svg.push(
        '<path d="' +
          linePath +
          '" fill="none" stroke="#2a78d6" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
      );

      series.forEach((point, index) => {
        const labelStep = Math.ceil(series.length / 7);

        if (index % labelStep === 0) {
          svg.push(
            '<text x="' +
              pointX(index).toFixed(1) +
              '" y="' +
              (height - 8) +
              '" text-anchor="middle" font-size="11" fill="#64748b">' +
              formatDate(point.date).slice(0, 5) +
              "</text>"
          );
        }

        svg.push(
          '<circle cx="' +
            pointX(index).toFixed(1) +
            '" cy="' +
            pointY(point.count).toFixed(1) +
            '" r="4" fill="#ffffff" stroke="#2a78d6" stroke-width="2" ' +
            'data-index="' +
            index +
            '"/>'
        );
      });

      svg.push("</svg>");
      container.innerHTML = svg.join("");

      const tooltip = this.ensureTooltip(container);
      const svgNode = container.querySelector("svg");

      svgNode.addEventListener("mousemove", (event) => {
        const rect = svgNode.getBoundingClientRect();
        const ratio = width / rect.width;
        const x = (event.clientX - rect.left) * ratio;
        const index = Math.max(
          0,
          Math.min(
            series.length - 1,
            Math.round((x - padding.left) / (stepX || 1))
          )
        );
        const point = series[index];

        tooltip.innerHTML =
          "<strong>" +
          formatDate(point.date) +
          "</strong><br />" +
          point.count +
          " interaction" +
          (point.count > 1 ? "s" : "");
        tooltip.style.left =
          (pointX(index) / ratio).toFixed(1) + "px";
        tooltip.style.top =
          (pointY(point.count) / ratio).toFixed(1) + "px";
        tooltip.classList.add("is-visible");
      });

      svgNode.addEventListener("mouseleave", () => {
        tooltip.classList.remove("is-visible");
      });
    },

    donut(container, items, options) {
      const config = options || {};
      container.innerHTML = "";

      const data = items.filter((item) => item.count > 0);

      if (!data.length) {
        container.innerHTML =
          '<p class="muted">Aucune donnée sur la période.</p>';
        return;
      }

      const total = data.reduce(
        (sum, item) => sum + item.count,
        0
      );
      const size = 190;
      const radius = 74;
      const thickness = 26;
      const center = size / 2;
      const circumference = 2 * Math.PI * radius;
      const gap = 2;

      let offset = 0;
      const segments = data.map((item, index) => {
        const fraction = item.count / total;
        const length = Math.max(
          fraction * circumference - gap,
          1
        );
        const segment =
          '<circle cx="' +
          center +
          '" cy="' +
          center +
          '" r="' +
          radius +
          '" fill="none" stroke="' +
          SERIES[index % SERIES.length] +
          '" stroke-width="' +
          thickness +
          '" stroke-dasharray="' +
          length.toFixed(2) +
          " " +
          (circumference - length).toFixed(2) +
          '" stroke-dashoffset="' +
          (-offset).toFixed(2) +
          '" data-index="' +
          index +
          '" transform="rotate(-90 ' +
          center +
          " " +
          center +
          ')"/>';

        offset += fraction * circumference;
        return segment;
      });

      const legend = data
        .map(
          (item, index) =>
            '<span class="legend__item">' +
            '<span class="legend__swatch" style="background:' +
            SERIES[index % SERIES.length] +
            '"></span>' +
            escapeHtml(item.label) +
            ' <span class="legend__value">' +
            item.count +
            " (" +
            Math.round((item.count / total) * 100) +
            "%)</span></span>"
        )
        .join("");

      container.innerHTML =
        '<div class="donut-wrap">' +
        '<svg viewBox="0 0 ' +
        size +
        " " +
        size +
        '" role="img" aria-label="' +
        escapeHtml(config.label || "Répartition") +
        '">' +
        segments.join("") +
        '<text x="' +
        center +
        '" y="' +
        (center - 4) +
        '" text-anchor="middle" class="donut-total">' +
        total +
        "</text>" +
        '<text x="' +
        center +
        '" y="' +
        (center + 16) +
        '" text-anchor="middle" class="donut-center">' +
        escapeHtml(config.unit || "échanges") +
        "</text>" +
        "</svg>" +
        '<div class="legend">' +
        legend +
        "</div>" +
        "</div>";

      const tooltip = this.ensureTooltip(container);

      $$("circle[data-index]", container).forEach((node) => {
        node.addEventListener("mousemove", (event) => {
          const item = data[Number(node.dataset.index)];
          const rect = container.getBoundingClientRect();

          tooltip.innerHTML =
            "<strong>" +
            escapeHtml(item.label) +
            "</strong><br />" +
            item.count +
            " — " +
            Math.round((item.count / total) * 100) +
            "%";
          tooltip.style.left =
            event.clientX - rect.left + "px";
          tooltip.style.top = event.clientY - rect.top + "px";
          tooltip.classList.add("is-visible");
        });

        node.addEventListener("mouseleave", () => {
          tooltip.classList.remove("is-visible");
        });
      });
    },

    bars(container, items) {
      container.innerHTML = "";

      if (!items.length) {
        container.innerHTML =
          '<p class="muted">Aucune donnée sur la période.</p>';
        return;
      }

      const maxValue = Math.max(
        ...items.map((item) => item.count)
      );

      container.innerHTML = items
        .map(
          (item) =>
            '<div class="bar-row">' +
            '<span class="bar-row__label" title="' +
            escapeHtml(item.label) +
            '">' +
            escapeHtml(item.label) +
            "</span>" +
            '<span class="bar-row__track">' +
            '<span class="bar-row__fill" style="width:' +
            Math.max(
              4,
              Math.round((item.count / maxValue) * 100)
            ) +
            '%"></span></span>' +
            '<span class="bar-row__value">' +
            item.count +
            "</span></div>"
        )
        .join("");
    },
  };

  /* ------------------------- Espace administration ------------------- */

  const Admin = {
    tab: "dashboard",
    // Verrou anti double-clic sur l'envoi d'une réponse.
    sendingResponse: false,
    // Confirmation à réafficher après le rechargement du dossier.
    pendingFeedback: null,

    token() {
      return sessionStorage.getItem("narsa-admin-token");
    },

    headers() {
      return { Authorization: "Bearer " + this.token() };
    },

    init() {
      $("#login-form").addEventListener("submit", (event) => {
        event.preventDefault();
        this.login(event.target);
      });

      $("#admin-nav").addEventListener("click", (event) => {
        const link = event.target.closest("[data-tab]");
        if (!link) return;

        event.preventDefault();

        if (link.dataset.tab === "logout") {
          this.logout();
          return;
        }

        this.switchTab(link.dataset.tab);
      });

      $("#dashboard-period").addEventListener("change", () => {
        this.loadDashboard();
      });

      $("#filter-apply").addEventListener("click", () => {
        this.loadComplaints();
      });

      $("#export-complaints").addEventListener("click", () => {
        this.download(
          "/admin/complaints/export?" +
            this.complaintFilters().toString(),
          "reclamations-narsa.csv"
        );
      });

      $("#export-conversations").addEventListener("click", () => {
        this.download(
          "/admin/conversations/export?limit=200",
          "conversations-narsa.csv"
        );
      });

      $("#topbar-logout").addEventListener("click", () => {
        this.logout();
      });

      $("#filter-search").addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          this.loadComplaints();
        }
      });
    },

    async login(form) {
      const errorBox = $("#login-error");
      errorBox.hidden = true;

      const payload = {};
      new FormData(form).forEach((value, key) => {
        payload[key] = String(value);
      });

      try {
        const data = await apiRequest("/admin/login", {
          method: "POST",
          body: payload,
        });

        sessionStorage.setItem(
          "narsa-admin-token",
          data.access_token
        );
        sessionStorage.setItem(
          "narsa-admin-name",
          data.display_name
        );

        form.reset();
        applyAuthState();
        navigate("dashboard");
      } catch (error) {
        errorBox.textContent = error.message;
        errorBox.hidden = false;
      }
    },

    logout() {
      sessionStorage.removeItem("narsa-admin-token");
      sessionStorage.removeItem("narsa-admin-name");
      this.tab = "dashboard";
      applyAuthState();
      navigate("accueil");
    },

    async download(path, filename) {
      try {
        const response = await fetch(API_BASE + path, {
          headers: this.headers(),
        });

        if (!response.ok) {
          throw new Error("Export impossible.");
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      } catch (error) {
        window.alert(error.message);
      }
    },

    switchTab(tab) {
      this.tab = tab;

      $$("#admin-nav a").forEach((link) => {
        link.classList.toggle(
          "is-active",
          link.dataset.tab === tab
        );
      });

      $("#admin-dashboard-tab").hidden = tab !== "dashboard";
      $("#admin-complaints-tab").hidden = tab !== "complaints";
      $("#admin-history-tab").hidden = tab !== "history";

      const titles = {
        dashboard: [
          "Dashboard IA",
          "Indicateurs de l'assistant intelligent",
        ],
        complaints: [
          "Réclamations",
          "Traitement des demandes citoyennes",
        ],
        history: [
          "Historique IA",
          "Derniers échanges enregistrés",
        ],
      };

      $("#admin-title").textContent = titles[tab][0];
      $("#admin-subtitle").textContent = titles[tab][1];

      this.refresh();
    },

    refresh() {
      if (!this.token()) return;

      $("#admin-user").textContent =
        sessionStorage.getItem("narsa-admin-name") || "Agent";

      if (this.tab === "dashboard") this.loadDashboard();
      if (this.tab === "complaints") this.loadComplaints();
      if (this.tab === "history") this.loadHistory();
    },

    handleAuthError(error) {
      if (/401|expir|jeton|Authentification/i.test(error.message)) {
        this.logout();
        return true;
      }
      return false;
    },

    async loadDashboard() {
      const days = $("#dashboard-period").value;

      try {
        const data = await apiRequest(
          "/admin/dashboard?days=" + days,
          { headers: this.headers() }
        );

        this.renderKpis(data.kpis);

        Charts.line(
          $("#chart-timeseries"),
          data.timeseries || []
        );

        Charts.donut(
          $("#chart-domains"),
          (data.domains || []).map((item) => ({
            label: domainLabel(item.domain),
            count: item.count,
          })),
          { label: "Répartition par domaines" }
        );

        Charts.bars(
          $("#chart-intents"),
          (data.top_intents || []).map((item) => ({
            label: item.intent.replace(/_/g, " "),
            count: item.count,
          }))
        );

        Charts.donut(
          $("#chart-answers"),
          (data.answer_types || []).map((item) => ({
            label: item.label,
            count: item.count,
          })),
          {
            label: "Analyse du chatbot",
            unit: "réponses",
          }
        );

        this.renderRecent(data.recent || []);
        this.loadInsights(days);
      } catch (error) {
        if (!this.handleAuthError(error)) {
          $("#kpis").innerHTML =
            '<p class="alert">' +
            escapeHtml(error.message) +
            "</p>";
        }
      }
    },

    async loadInsights(days) {
      try {
        const data = await apiRequest(
          "/admin/insights?days=" + days,
          { headers: this.headers() }
        );

        $("#alerts").innerHTML = (data.alerts || [])
          .map(
            (alert) =>
              '<div class="alert-card alert-card--' +
              escapeHtml(alert.level) +
              '"><span class="alert-card__dot"></span>' +
              "<div><strong>" +
              escapeHtml(alert.title) +
              "</strong><p>" +
              escapeHtml(alert.detail) +
              "</p></div></div>"
          )
          .join("");

        const spots = data.blind_spots || [];

        $("#blind-spots").innerHTML = spots.length
          ? spots
              .map(
                (spot) =>
                  '<div class="blind-spot">' +
                  '<div class="blind-spot__head">' +
                  '<span class="blind-spot__question">' +
                  escapeHtml(spot.question) +
                  "</span>" +
                  '<span class="blind-spot__count">' +
                  spot.occurrences +
                  (spot.occurrences > 1
                    ? " questions"
                    : " question") +
                  "</span></div>" +
                  '<p class="blind-spot__meta">Domaine(s) : ' +
                  escapeHtml(
                    (spot.domains || [])
                      .map(domainLabel)
                      .join(", ") || "non déterminé"
                  ) +
                  " — dernière occurrence le " +
                  formatDate(spot.last_seen, true) +
                  "</p>" +
                  ((spot.examples || []).length
                    ? '<ul class="blind-spot__examples">' +
                      spot.examples
                        .map(
                          (example) =>
                            "<li>" +
                            escapeHtml(example) +
                            "</li>"
                        )
                        .join("") +
                      "</ul>"
                    : "") +
                  "</div>"
              )
              .join("")
          : '<p class="muted">Aucune question sans réponse sur la période : le corpus documentaire couvre les demandes reçues.</p>';
      } catch (error) {
        this.handleAuthError(error);
      }
    },

    renderKpis(kpis) {
      const definitions = [
        ["conversations", "Conversations", (v) => v],
        ["interactions", "Interactions", (v) => v],
        [
          "avg_confidence",
          "Confiance moyenne",
          (v) => Number(v).toFixed(2),
        ],
        [
          "avg_time_ms",
          "Temps moyen réponse",
          (v) => formatDuration(v),
        ],
        ["clarifications", "Classification IA", (v) => v],
        ["low_confidence", "Faites sans humains", (v) => v],
      ];

      $("#kpis").innerHTML = definitions
        .map((definition) => {
          const key = definition[0];
          const entry = kpis[key] || { value: 0, change: null };
          const change = entry.change;

          let changeClass = "flat";
          let changeText = "—";

          if (change !== null && change !== undefined) {
            changeClass =
              change > 0 ? "up" : change < 0 ? "down" : "flat";
            changeText =
              (change > 0 ? "+" : "") + change + " %";
          }

          return (
            '<div class="kpi">' +
            '<div class="kpi__label">' +
            definition[1] +
            "</div>" +
            '<div class="kpi__value">' +
            definition[2](entry.value) +
            "</div>" +
            '<div class="kpi__change kpi__change--' +
            changeClass +
            '">' +
            changeText +
            "</div></div>"
          );
        })
        .join("");
    },

    renderRecent(rows) {
      const body = $("#recent-table tbody");

      if (!rows.length) {
        body.innerHTML =
          '<tr><td colspan="6" class="table__empty">Aucun échange enregistré pour le moment.</td></tr>';
        return;
      }

      body.innerHTML = rows
        .map(
          (row) =>
            "<tr><td>" +
            escapeHtml(row.question) +
            "</td><td>" +
            escapeHtml(domainLabel(row.domain)) +
            '</td><td class="mono">' +
            escapeHtml(row.intent) +
            "</td><td>" +
            Number(row.confidence).toFixed(2) +
            "</td><td>" +
            formatDuration(row.time_ms) +
            "</td><td>" +
            formatDate(row.created_at, true) +
            "</td></tr>"
        )
        .join("");
    },

    async loadHistory() {
      try {
        const data = await apiRequest(
          "/admin/conversations?limit=50",
          { headers: this.headers() }
        );

        const body = $("#history-table tbody");
        const rows = data.items || [];

        if (!rows.length) {
          body.innerHTML =
            '<tr><td colspan="6" class="table__empty">Aucun échange enregistré.</td></tr>';
          return;
        }

        body.innerHTML = rows
          .map(
            (row) =>
              "<tr><td>" +
              escapeHtml(row.question) +
              "</td><td>" +
              escapeHtml(domainLabel(row.domain)) +
              '</td><td class="mono">' +
              escapeHtml(row.intent) +
              "</td><td>" +
              Number(row.confidence).toFixed(2) +
              '</td><td class="mono">' +
              escapeHtml(row.status) +
              "</td><td>" +
              formatDate(row.created_at, true) +
              "</td></tr>"
          )
          .join("");
      } catch (error) {
        this.handleAuthError(error);
      }
    },

    complaintFilters() {
      const params = new URLSearchParams();
      const status = $("#filter-status").value;
      const priority = $("#filter-priority").value;
      const search = $("#filter-search").value.trim();

      if (status) params.set("status", status);
      if (priority) params.set("priority", priority);
      if (search) params.set("search", search);

      return params;
    },

    async loadComplaintKpis() {
      try {
        const data = await apiRequest(
          "/admin/dashboard?days=90",
          { headers: this.headers() }
        );

        const stats = data.complaints || {};

        $("#complaints-kpis").innerHTML = (
          stats.by_status || []
        )
          .map(
            (item) =>
              '<div class="kpi">' +
              '<div class="kpi__label">' +
              escapeHtml(item.label) +
              "</div>" +
              '<div class="kpi__value">' +
              item.count +
              "</div></div>"
          )
          .join("");
      } catch (error) {
        this.handleAuthError(error);
      }
    },

    async loadComplaints() {
      const params = this.complaintFilters();

      this.loadComplaintKpis();

      try {
        const data = await apiRequest(
          "/admin/complaints?" + params.toString(),
          { headers: this.headers() }
        );

        $("#complaints-count").textContent =
          data.total + " réclamation(s)";

        const body = $("#complaints-table tbody");

        if (!data.items.length) {
          body.innerHTML =
            '<tr><td colspan="8" class="table__empty">Aucune réclamation ne correspond à ces filtres.</td></tr>';
          $("#complaint-detail").innerHTML = "";
          return;
        }

        body.innerHTML = data.items
          .map(
            (item) =>
              '<tr><td class="mono">' +
              escapeHtml(item.reference) +
              "</td><td>" +
              escapeHtml(item.full_name) +
              "<br /><span class='muted'>" +
              escapeHtml(item.email) +
              "</span></td><td>" +
              escapeHtml(item.category_label) +
              "</td><td>" +
              escapeHtml(item.subject) +
              "</td><td>" +
              priorityBadge(item.priority, item.priority_label) +
              "</td><td>" +
              statusBadge(item.status, item.status_label) +
              "</td><td>" +
              formatDate(item.created_at) +
              '</td><td><button class="btn btn--ghost btn--sm" data-reference="' +
              escapeHtml(item.reference) +
              '">Ouvrir</button></td></tr>'
          )
          .join("");

        $$("#complaints-table [data-reference]").forEach(
          (button) => {
            button.addEventListener("click", () => {
              this.openComplaint(button.dataset.reference);
            });
          }
        );
      } catch (error) {
        if (!this.handleAuthError(error)) {
          $("#complaints-count").textContent = error.message;
        }
      }
    },

    responseFeedback(saved) {
      // La réponse est déjà en base quand on arrive ici : le
      // message distingue toujours l'enregistrement (réussi)
      // de la notification email (qui peut, elle, échouer).
      const notification = (saved && saved.notification) || null;
      const base = "Réponse enregistrée et visible dans le suivi citoyen.";

      if (!notification) {
        return { message: base, ok: true };
      }

      if (notification.status === "sent") {
        return {
          message:
            base + " Email de notification envoyé à " + notification.to + ".",
          ok: true,
        };
      }

      if (notification.status === "console") {
        return {
          message:
            base +
            " Aucun serveur SMTP configuré : l'email a été écrit " +
            "dans la console du serveur (SMTP_HOST manquant).",
          ok: false,
        };
      }

      if (notification.status === "disabled") {
        return {
          message:
            base +
            " Notifications email désactivées (EMAIL_ENABLED=false).",
          ok: false,
        };
      }

      return {
        message:
          base +
          " En revanche l'email n'a PAS pu être envoyé : " +
          (notification.detail || notification.error || "erreur SMTP") +
          " Consultez le journal du serveur pour le détail.",
        ok: false,
      };
    },

    async openComplaint(reference) {
      const panel = $("#complaint-detail");

      try {
        const data = await apiRequest(
          "/admin/complaints/" + encodeURIComponent(reference),
          { headers: this.headers() }
        );

        const field = (label, value) =>
          '<div class="case-field"><span>' +
          escapeHtml(label) +
          "</span><strong>" +
          escapeHtml(value || "—") +
          "</strong></div>";

        panel.innerHTML =
          '<div class="card">' +
          '<div class="case-head">' +
          "<h3>Réclamation #" +
          escapeHtml(data.reference) +
          "</h3>" +
          "<div style='display:flex;gap:10px;flex-wrap:wrap'>" +
          statusBadge(data.status, data.status_label) +
          priorityBadge(data.priority, data.priority_label) +
          '<button class="btn btn--ghost btn--sm" type="button" id="case-print">Imprimer</button>' +
          "</div></div>" +
          '<p class="muted">Soumise le ' +
          formatDate(data.created_at, true) +
          "</p>" +
          '<div class="case-grid">' +

          '<section class="case-block">' +
          "<h4>Informations du citoyen</h4>" +
          field("Nom complet", data.full_name) +
          field("Email", data.email) +
          field("Téléphone", data.phone) +
          field("Ville", data.city) +
          "<h4 style='margin-top:16px'>Objet de la réclamation</h4>" +
          "<p>" +
          escapeHtml(data.subject) +
          "</p>" +
          "<h4 style='margin-top:16px'>Message du citoyen</h4>" +
          '<p class="muted">' +
          escapeHtml(data.message) +
          "</p></section>" +

          '<section class="case-block">' +
          "<h4>Suivi du dossier</h4>" +
          '<ul class="track-timeline">' +
          (data.events || [])
            .map(
              (event, index, all) =>
                '<li class="track-step' +
                (index === all.length - 1
                  ? " track-step--current"
                  : " track-step--done") +
                '"><span class="track-step__icon">' +
                '<svg class="ico"><use href="#i-check"/></svg></span>' +
                '<div class="track-step__body"><strong>' +
                escapeHtml(event.status_label) +
                "</strong>" +
                (event.note
                  ? "<p>" + escapeHtml(event.note) + "</p>"
                  : "") +
                '<p class="muted">' +
                formatDate(event.created_at, true) +
                " — " +
                escapeHtml(event.author) +
                "</p></div></li>"
            )
            .join("") +
          "</ul></section>" +

          '<section class="case-block">' +
          "<h4>Statut du dossier</h4>" +
          field(
            "Sujet détecté",
            data.detected_intent
              ? data.detected_intent.replace(/_/g, " ")
              : "non déterminé"
          ) +
          field("Priorité estimée", data.priority_label) +
          '<form class="form" id="status-form" style="margin-top:14px">' +
          "<label>Statut actuel" +
          '<select name="status">' +
          '<option value="recue">Reçue</option>' +
          '<option value="en_cours">En cours de traitement</option>' +
          '<option value="traitee">Traitée</option>' +
          '<option value="rejetee">Rejetée</option>' +
          "</select></label>" +
          "<label>Note interne communiquée au citoyen" +
          '<textarea name="note" rows="2" placeholder="Ajouter une note..."></textarea></label>' +
          '<button class="btn btn--ghost btn--block" type="submit">Mettre à jour le statut</button>' +
          "</form>" +
          "<h4 style='margin-top:18px'>Réponse officielle</h4>" +
          (data.official_response
            ? '<div class="official-response official-response--compact">' +
              renderRichText(data.official_response) +
              '<p class="official-response__meta">Transmise le ' +
              formatDate(data.responded_at, true) +
              (data.responded_by
                ? " par " + escapeHtml(data.responded_by)
                : "") +
              "</p></div>"
            : "") +
          '<form class="form" id="response-form">' +
          "<label>" +
          (data.official_response
            ? "Modifier la réponse envoyée au citoyen"
            : "Rédiger la réponse envoyée au citoyen") +
          '<textarea name="response" rows="7" placeholder="Bonjour, après vérification de votre dossier...">' +
          // La réponse déjà enregistrée est rechargée dans le
          // champ : l'agent la relit et la corrige au lieu de
          // la ressaisir.
          escapeHtml(data.official_response || "") +
          "</textarea></label>" +
          "<label>Statut après envoi" +
          '<select name="status">' +
          '<option value="traitee">Traitée</option>' +
          '<option value="rejetee">Rejetée</option>' +
          '<option value="en_cours">En cours de traitement</option>' +
          "</select></label>" +
          '<button class="btn btn--primary btn--block" type="submit" id="response-submit">' +
          (data.official_response
            ? "Mettre à jour la réponse"
            : "Envoyer la réponse au citoyen") +
          "</button>" +
          '<p class="alert" id="response-feedback" hidden></p>' +
          '<p class="muted">La réponse est conservée dans la plateforme ; un email prévient le citoyen.</p>' +
          "</form></section>" +

          "</div></div>";

        const printButton = $("#case-print", panel);

        if (printButton) {
          printButton.addEventListener("click", () =>
            window.print()
          );
        }

        const responseForm = $("#response-form", panel);

        if (responseForm) {
          const submitButton = $("#response-submit", responseForm);
          const feedback = $("#response-feedback", responseForm);
          const initialLabel = submitButton.textContent;

          // Le statut proposé après envoi part du statut réel
          // du dossier quand il est déjà tranché.
          if (
            ["traitee", "rejetee", "en_cours"].includes(data.status)
          ) {
            responseForm.status.value = data.status;
          }

          const showFeedback = (message, ok) => {
            feedback.textContent = message;
            feedback.classList.toggle("alert--ok", Boolean(ok));
            feedback.hidden = false;
          };

          responseForm.addEventListener(
            "submit",
            async (event) => {
              event.preventDefault();

              // Garde anti double-clic : une soumission déjà
              // en cours ne peut pas être relancée.
              if (this.sendingResponse) return;

              const text = responseForm.response.value.trim();

              if (text.length < 10) {
                showFeedback(
                  "La réponse doit contenir au moins 10 caractères.",
                  false
                );
                return;
              }

              this.sendingResponse = true;
              submitButton.disabled = true;
              submitButton.textContent = "Envoi en cours...";
              feedback.hidden = true;

              try {
                const saved = await apiRequest(
                  "/admin/complaints/" +
                    encodeURIComponent(data.reference) +
                    "/response",
                  {
                    method: "POST",
                    headers: this.headers(),
                    body: {
                      response: text,
                      status: responseForm.status.value,
                    },
                  }
                );

                // La réponse est enregistrée : le résultat de
                // la notification email est signalé à part pour
                // qu'un problème SMTP reste diagnosticable.
                this.pendingFeedback = this.responseFeedback(saved);

                await this.loadComplaints();
                await this.openComplaint(data.reference);
              } catch (error) {
                if (!this.handleAuthError(error)) {
                  showFeedback(
                    "Échec de l'enregistrement : " + error.message,
                    false
                  );
                }
              } finally {
                this.sendingResponse = false;
                submitButton.disabled = false;
                submitButton.textContent = initialLabel;
              }
            }
          );

          // Confirmation affichée après le rechargement du
          // dossier, pour qu'elle survive au re-rendu du panneau.
          if (this.pendingFeedback) {
            showFeedback(
              this.pendingFeedback.message,
              this.pendingFeedback.ok
            );
            this.pendingFeedback = null;
          }
        }

        const statusForm = $("#status-form", panel);
        statusForm.status.value = data.status;

        statusForm.addEventListener("submit", async (event) => {
          event.preventDefault();

          const payload = {
            status: statusForm.status.value,
          };
          const note = statusForm.note.value.trim();
          if (note) payload.note = note;

          try {
            await apiRequest(
              "/admin/complaints/" +
                encodeURIComponent(data.reference),
              {
                method: "PATCH",
                headers: this.headers(),
                body: payload,
              }
            );

            await this.loadComplaints();
            await this.openComplaint(data.reference);
          } catch (error) {
            if (!this.handleAuthError(error)) {
              window.alert(error.message);
            }
          }
        });

        panel.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        this.handleAuthError(error);
      }
    },
  };

  /* ------------------------------ Démarrage -------------------------- */

  function init() {
    Chat.init();
    Complaints.init();
    Admin.init();
    Journey.init();
    SourceViewer.init();

    $("#burger").addEventListener("click", () => {
      const bar = $("#topbar");
      const open = bar.classList.toggle("is-open");
      $("#burger").setAttribute("aria-expanded", String(open));
    });

    window.addEventListener("hashchange", showRoute);
    showRoute();

    apiRequest("/stats")
      .then((data) => {
        const format = (value) =>
          Number(value || 0).toLocaleString("fr-FR");

        $("#figure-complaints").textContent = format(
          data.complaints
        );
        $("#figure-conversations").textContent = format(
          data.conversations
        );
        $("#figure-sources").textContent =
          (data.sourced_rate || 0) + " %";
      })
      .catch(() => {});

    apiRequest("/health")
      .then((data) => {
        $("#api-status").textContent = data.generation_enabled
          ? "API v" + data.version + " — génération Gemini active"
          : "API v" +
            data.version +
            " — mode documentaire (sans génération)";
      })
      .catch(() => {
        $("#api-status").textContent =
          "API NARSA injoignable — démarrez le serveur.";
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
